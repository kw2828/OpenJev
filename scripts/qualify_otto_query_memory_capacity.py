"""Bounded synthetic query-memory capacity, with closed collection metadata only.

Collection payloads are authenticated as opaque bytes. Only episode identity and
length metadata determine planned chunk counts; no empirical array, checkpoint,
teacher or simulator is decoded or invoked. A completed capacity run is only a
heuristic admission estimate, never a training completion or efficacy result.
"""
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
SELF = "scripts/qualify_otto_query_memory_capacity.py"
TEST = "tests/test_qualify_otto_query_memory_capacity.py"
CORE = "src/openjev/research/otto_query_memory_training.py"
CORE_TEST = "tests/test_otto_query_memory_training.py"
COLLECTOR = "scripts/collect_otto_query_memory.py"
COLLECTOR_PIN = "e1cd73b7e2389c208955ce5dd46e7f7238444482d5ae06d7ee73fc640a579c8f"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
PROTOCOL = "research/otto-query-memory-protocol.md"
INTERPRETER = ".venv/bin/python"
VERSION = "otto-query-memory-capacity-v1"
FAMILIES = ("ordinary", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
SEEDS = (309000001, 309000002, 309000003)
COMPONENTS = {SELF, TEST, CORE, CORE_TEST}
NEW_SOURCES = COMPONENTS | {COLLECTOR, CLOCK, SUPERVISOR, PROTOCOL}
ROLES = {"collection_plan", "collection_receipt", "collection_terminal", "engineering"}
PAYLOADS = {"started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"}
LIMITS = {"seconds": 240, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
CONFIGURATION = {"families": list(FAMILIES), "batch": 6, "length": 256, "chunk": 32, "period": 4,
    "synthetic_seed": 1001, "model_seed": 1001, "key_dim": 8, "train_episodes": 54,
    "fit_seeds": list(SEEDS), "pretraining_epochs": 80, "branch_epochs": 40,
    "branch_orders": "PCG64(fit_seed) restarted in every branch",
    "objective": "full_forecast_aux", "learning_rate": .003, "weight_decay": 0., "clip_norm": 5.,
    "projection_multiplier": 2., "fixed_overhead_seconds": 600.,
    "threshold_seconds": 16200., "training_cap_seconds": 21600,
    "scope": "synthetic timing plus authenticated TRAIN lengths only; heuristic admission, not performance"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "regular contained input")
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
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


def load(value, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / value)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_record():
    return {"python": sys.version, "executable": sys.executable,
            "distributions": dict(sorted((x.metadata["Name"], x.version)
                                          for x in importlib.metadata.distributions()))}


def original_collection_process(inputs, receipt, plan):
    """Authenticate the real original parent and the worker's exact launch joins."""
    terminal = read(inputs["collection_terminal"]["path"])
    directory = regular(inputs["collection_receipt"]["path"]).parent
    started = read(directory / "started.json")
    request = started["request"]
    launch_path = regular(request["supervision"])
    launch = read(launch_path)
    require(descriptor(launch_path)["sha256"] == receipt["supervision_sha256"]
            and started["launch"] == launch and started["started_ns"] == receipt["started_ns"],
            "original collection worker/launch identity")
    require(terminal["status"] == "completed" and type(terminal["returncode"]) is int
            and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["errors"] == []
            and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["timing_available"] is True and terminal["cap_seconds"] == 7200
            and terminal["clock_source_sha256"] == CLOCK_PIN and terminal["watchdog_sha256"] == SUPERVISOR_PIN
            and terminal["deadline_ns"] == terminal["started_ns"] + 7200 * 10**9
            and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
            <= terminal["finished_ns"] <= terminal["deadline_ns"], "genuine successful original collection parent")
    fields = ("command", "cwd", "pid", "pgid", "parent_pid", "started_ns", "deadline_ns", "cap_seconds",
              "clock_backend", "clock_source_sha256", "watchdog_sha256")
    require(all(terminal[key] == launch[key] for key in fields) and Path(terminal["cwd"]) == ROOT
            and terminal["pid"] == terminal["pgid"] and terminal["pid"] != terminal["parent_pid"],
            "collection parent/launch process joins")
    command = list(launch["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    expected = {"--plan": str(regular(inputs["collection_plan"]["path"])),
                "--plan-sha256": inputs["collection_plan"]["sha256"],
                "--supervision": str(launch_path), "--output": str(directory)}
    require(command[:3] == [str(ROOT / ".venv-otto-released-native/bin/python"), str(ROOT / COLLECTOR), "run"]
            and len(command) == 11 and len(set(command[3::2])) == 4
            and dict(zip(command[3::2], command[4::2], strict=True)) == expected,
            "collection exact command and artifact joins")
    require(request == {"mode": "run", **{key[2:].replace("-", "_"): value for key, value in expected.items()}}
            and receipt["limits"] == plan["limits"], "collection started request and limits")


def authenticate_collection(inputs, collector):
    """Metadata-only admission; every payload is hashed, none is array-decoded."""
    plan = read(inputs["collection_plan"]["path"])
    require(plan["version"] == collector.VERSION and plan["status"] == "frozen_before_collection"
            and plan["configuration"] == collector.CONFIGURATION and plan["limits"] == collector.LIMITS
            and plan["call_caps"] == collector.CALL_CAPS and plan["cohort"] == collector.cohort()
            and set(plan["payloads"]) == collector.PAYLOADS, "fixed complete108-path collection plan")
    # Do not invoke the collector's native authenticator: that function admits a
    # live TF interpreter. This consumer owns a separate pinned Torch runtime.
    # The original collector already performed native admission; verify its
    # complete saved evidence and unchanged bytes without loading native code.
    sources = dict(plan["sources"])
    require(bool(sources), "nonempty original collection source closure")
    for name, digest in sources.items():
        require(descriptor(name)["sha256"] == digest, "unchanged inherited collection source")
    for section in ("inputs", "native_inputs"):
        require(isinstance(plan[section], dict), "collection evidence descriptor map")
        for record in plan[section].values():
            require(set(record) == {"path", "sha256", "bytes"}
                    and descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")},
                    "unchanged inherited collection input bytes")
    receipt = read(inputs["collection_receipt"]["path"])
    require(receipt["version"] == collector.VERSION and receipt["status"] == "completed"
            and receipt["complete"] is True and receipt["plan_sha256"] == inputs["collection_plan"]["sha256"]
            and receipt["sources"] == sources and receipt["inputs"] == plan["inputs"]
            and receipt["native_inputs"] == plan["native_inputs"] and receipt["training_updates"] == 0
            and receipt["completed_episodes"] == 108
            and [receipt[k] for k in ("train_episodes", "dev_episodes", "test_episodes")] == [54, 18, 36]
            and receipt["pending"] == [] and receipt["pending_emission"] is None
            and receipt["pending_episode"] is None and receipt["pending_action"] is None
            and receipt["requires_successful_original_supervisor"] is True
            and not receipt.get("cleanup_errors") and receipt["peak_rss_bytes"] <= collector.LIMITS["rss_bytes"],
            "genuine complete collection worker without pending work")
    directory = regular(inputs["collection_receipt"]["path"]).parent
    require(set(receipt["files"]) == collector.PAYLOADS
            and {p.name for p in directory.iterdir()} == collector.PAYLOADS | {"receipt.json"},
            "exact17-payload collection closure")
    for name, expected in receipt["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == expected, "opaque collection payload hash")
    require(sum(item["bytes"] for item in receipt["files"].values()) <= collector.LIMITS["output_bytes"],
            "closed collection output bound")
    original_collection_process(inputs, receipt, plan)
    runtime = read(directory / "runtime.json")
    require(runtime["executable"] == plan["runtime"]["python_executable"]
            == str(ROOT / ".venv-otto-released-native/bin/python")
            and runtime["python"].split()[0] == plan["runtime"]["python_version"]
            and runtime["all_distributions"] == plan["runtime"]["all_distributions"],
            "saved native runtime equals original collection declaration")
    calls = receipt["calls"]
    require(set(calls) == set(collector.CALL_CAPS) and all(
        type(row["attempted"]) is int and row["attempted"] == row["returned"]
        and 0 <= row["attempted"] <= collector.CALL_CAPS[name]
        for name, row in calls.items()), "completed bounded collection operation counts")
    # The first access to episode content occurs only after all closure checks.
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    roster = collector.cohort()
    require(len(rows) == len(roster) == 108 and all(
        {key: row[key] for key in identity} == identity for row, identity in zip(rows, roster, strict=True)),
        "exact complete ordered collection episode metadata")
    # DEV/TEST lengths and outcome fields do not enter this capacity calculation.
    total = calls["teacher_score"]["returned"]
    for name in ("teacher_score", "tensorflow_value", "native_step", "actor_update", "analytic_score", "feature_build"):
        require(calls[name]["returned"] == total, "one collection operation per state")
    for name in ("native_reset", "actor_construction", "backend_binding"):
        require(calls[name]["returned"] == 108, "one collection operation per path")
    for name in ("tensorflow_construction", "tensorflow_build", "tensorflow_load"):
        require(calls[name]["returned"] == 1, "one original model setup")
    lengths = [row["steps"] for row in rows if row["stage"] == "train"]
    require(len(lengths) == 54 and all(type(n) is int and 1 <= n <= 2188 for n in lengths),
            "all54 bounded positive TRAIN lengths")
    metadata = {"path": str((directory / "episodes.jsonl").relative_to(ROOT)),
                **descriptor(directory / "episodes.jsonl"), "train_lengths": lengths,
                "scope": "only original TRAIN episode lengths used; no empirical array/checkpoint decode"}
    return dict(sources), metadata


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact capacity input roles")
    for record in inputs.values():
        require(set(record) == {"path", "sha256", "bytes"}
                and descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")},
                "capacity input descriptor")
    require(descriptor(COLLECTOR)["sha256"] == COLLECTOR_PIN, "qualified collector source before import")
    collector = load(COLLECTOR, "_query_memory_capacity_collection")
    sources, metadata = authenticate_collection(inputs, collector)
    for name in NEW_SOURCES:
        digest = descriptor(name)["sha256"]
        require(name not in sources or sources[name] == digest, "new source cannot replace inherited bytes")
        sources[name] = digest
    require(sources[CLOCK] == CLOCK_PIN and sources[SUPERVISOR] == SUPERVISOR_PIN,
            "qualified original clock and supervisor")
    authenticate_engineering(inputs["engineering"]["path"], sources)
    return sources, metadata


def authenticate_engineering(path, sources):
    """Current descriptor receipt, exact logs, and only pytest-temp exclusion."""
    engineering = read(path)
    require(engineering["status"] == "passed" and isinstance(engineering["source_after"], dict)
            and engineering["source_before"] == engineering["source_after"]
            and COMPONENTS <= set(engineering["source_after"])
            and isinstance(engineering["commands"], list) and engineering["commands"]
            and all(type(command["returncode"]) is int and command["returncode"] == 0
                    and command["timed_out"] is False and command["reaped"] is True
                    for command in engineering["commands"]), "all current capacity/core qualifications")
    for name, expected in engineering["source_after"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and isinstance(expected, dict) and set(expected) == {"sha256", "bytes"}
                and descriptor(name) == expected, "qualified current source descriptor")
        require(name not in sources or sources[name] == expected["sha256"],
                "qualified source cannot replace inherited bytes")
        sources[name] = expected["sha256"]
    directory = regular(path).parent
    names = set()
    for entry in directory.iterdir():
        if entry.name == "pytest-temp":
            require(entry.is_dir() and not entry.is_symlink(), "genuine pytest fixture directory")
            continue
        require(entry.is_file() and not entry.is_symlink(), "only declared regular qualification files")
        names.add(entry.name)
    require(isinstance(engineering["files"], dict)
            and names == set(engineering["files"]) | {"receipt.json"}, "exact qualification log closure")
    for name, expected in engineering["files"].items():
        require(Path(name).name == name and set(expected) == {"sha256", "bytes"}
                and descriptor(directory / name) == expected, "qualification log descriptor")
    for command in engineering["commands"]:
        require(command["log"] in engineering["files"]
                and {key: command[key] for key in ("sha256", "bytes")}
                == engineering["files"][command["log"]], "command joins authenticated log")


def planned_chunks(lengths, *, check=lambda: None):
    """Exact PCG64 batch maxima, preserving each independent branch restart."""
    require(isinstance(lengths, list) and len(lengths) == 54
            and all(type(n) is int and 1 <= n <= 2188 for n in lengths), "exact54 bounded TRAIN lengths")
    import numpy as np

    result = dict.fromkeys(FAMILIES, 0)
    for seed in SEEDS:
        for epochs, families in ((80, ("ordinary",)), (40, FAMILIES)):
            generator = np.random.Generator(np.random.PCG64(seed))
            count = 0
            for _ in range(epochs):
                check()
                order = generator.permutation(54).tolist()
                for low in range(0, 54, 6):
                    count += (max(lengths[index] for index in order[low:low + 6]) + 31) // 32
            for family in families:
                result[family] += count
    return result


def capacity_summary(rows, chunks):
    require(isinstance(rows, list) and [row["family"] for row in rows] == list(FAMILIES)
            and set(chunks) == set(FAMILIES)
            and all(type(n) is int and n > 0 for n in chunks.values()), "all five synthetic families and planned counts")
    require(all(type(row["batch_seconds"]) in (int, float) and math.isfinite(row["batch_seconds"])
                and row["batch_seconds"] > 0 and row["forward_chunks"] == 8 for row in rows),
            "positive full-batch synthetic timing with eight chunks")
    projected = 2 * math.fsum(row["batch_seconds"] / 8 * chunks[row["family"]] for row in rows) + 600
    require(math.isfinite(projected), "finite projected training time")
    return {"version": VERSION, "families": rows, "planned_chunks": chunks,
            "projected_seconds": projected, "threshold_seconds": 16200., "training_cap_seconds": 21600,
            "admitted": projected <= 16200., "technical_ready": True,
            "scope": "heuristic2x measured batch/chunk extrapolation plus600s; hard training cap remains21600s",
            "empirical_array_decodes": 0, "empirical_checkpoint_decodes": 0,
            "teacher_calls": 0, "native_calls": 0}


def freeze(args):
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role))
        item = descriptor(path)
        require(item["sha256"] == getattr(args, role + "_sha256"), "external capacity input pin")
        inputs[role] = {"path": str(path), **item}
    sources, metadata = authenticate_inputs(inputs)
    plan = {"version": VERSION, "status": "frozen_before_synthetic_work", "configuration": CONFIGURATION,
            "limits": LIMITS, "sources": sources, "inputs": inputs, "runtime": runtime_record(),
            "collection_metadata": metadata, "planned_chunks": planned_chunks(metadata["train_lengths"]),
            "payloads": sorted(PAYLOADS)}
    write(args.output, plan)
    print(json.dumps({"plan": str(args.output), **descriptor(args.output)}), flush=True)


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.launch = self.start = self.plan = None
        self.sequence = 0
        self.receipt = {"version": VERSION, "status": "started", "complete": False, "admitted": False,
            "pending": None, "pending_emission": None, "completed_families": [], "completed_family_count": 0,
            "optimizer_updates": 0,
            "empirical_array_decodes": 0, "empirical_checkpoint_decodes": 0, "teacher_calls": 0, "native_calls": 0}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original capacity deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "capacity RSS/output cap with failure reserve")

    def event(self, record):
        self.check()
        self.receipt["pending_emission"] = record
        with (self.out / "work.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.receipt["pending_emission"] = None

    def call(self, family, operation, function):
        self.check()
        require(self.receipt["pending"] is None, "no overlapping capacity operations")
        self.sequence += 1
        record = {"call_id": self.sequence, "family": family, "operation": operation}
        self.receipt["pending"] = dict(record)
        self.event({"event": "attempt", **record})
        tick = self.clock.now_ns()
        result = function()
        elapsed = (self.clock.now_ns() - tick) / 1e9
        require(elapsed >= 0, "nonnegative original-clock operation time")
        self.event({"event": "return", **record, "seconds": elapsed})
        self.receipt["pending"] = None
        self.check()
        return result, elapsed

    def admit(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN, "clock pin before import")
        self.clock = load(CLOCK, "_query_memory_capacity_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv]
                and command[:3] == [str(ROOT / INTERPRETER), str(ROOT / SELF), "run"]
                and self.launch["pid"] == os.getpid() and self.launch["pgid"] == os.getpgrp()
                and self.launch["parent_pid"] == os.getppid() and os.getpid() != os.getppid()
                and Path(self.launch["cwd"]) == Path.cwd() == ROOT
                and self.launch["cap_seconds"] == LIMITS["seconds"]
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["clock_source_sha256"] == CLOCK_PIN
                and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9,
                "canonical original240s capacity supervisor")
        self.check()
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external capacity plan pin")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_synthetic_work"
                and self.plan["configuration"] == CONFIGURATION and self.plan["limits"] == LIMITS
                and self.plan["runtime"] == runtime_record() and set(self.plan["payloads"]) == PAYLOADS,
                "fixed capacity allocation and exact runtime libraries")
        sources, metadata = authenticate_inputs(self.plan["inputs"])
        require(sources == self.plan["sources"] and metadata == self.plan["collection_metadata"],
                "unchanged collection and source closure")
        require(all(os.environ.get(name) == "1" for name in THREADS), "single-thread numerical environment")
        require("torch" not in sys.modules, "metadata admission precedes Torch import")
        require(planned_chunks(metadata["train_lengths"], check=self.check) == self.plan["planned_chunks"],
                "exact frozen original-length chunk forecast")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=sources, inputs=self.plan["inputs"],
            limits=LIMITS, supervision_sha256=descriptor(self.args.supervision)["sha256"])
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch,
            "request": {key: str(value) for key, value in vars(self.args).items()}})

    def synthetic(self):
        np = self.np
        generator = np.random.Generator(np.random.PCG64(1001))
        batch, length = CONFIGURATION["batch"], CONFIGURATION["length"]
        steps = np.tile(np.arange(length, dtype=np.int64), batch)
        total = batch * length
        features = generator.uniform(-.25, .25, (total, 31)).astype(np.float32)
        features[:, 15] = (steps / 2188).astype(np.float32)
        features[:, 16] = ((steps % 4) / 2188).astype(np.float32)
        features[:, 17] = 1
        legal = np.ones((total, 4), np.bool_)
        legal[::7, 0] = False
        features[:, 2:6] = legal.astype(np.float32)
        flat = {"features": features, "raw_q": generator.uniform(28, 36, (total, 4)).astype(np.float32),
                "legal": legal, "actions": np.ones(total, np.int64), "correction": steps % 4 == 0,
                "episode_offsets": np.arange(batch + 1, dtype=np.int64) * length}
        identities = [{"stage": "train", "episode_id": f"synthetic-{i}", "episode_index": i,
                       "seed": 1001, "case": i // 3, "regime": "lambda3",
                       "arm": ("analytic", "neural", "period4_hold")[i % 3]} for i in range(batch)]
        projected = self.data.project_census(flat, identities, query_period=4, expected_stage="train")
        metadata = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in flat.items()}
        write(self.out / "synthetic.json", {"configuration": CONFIGURATION, "arrays": metadata,
            "generator": "NumPy PCG64(1001), new fabricated arrays only", "episode_count": 6,
            "nonquery_rows": 6 * 192, "prior_rows": 6 * 63, "empirical_values_used": False,
            "normalization": "six-episode synthetic denominator; whole-batch mean equals54/6 training scaling"})
        return projected

    def one_family(self, family, data):
        torch = self.torch
        mode = "none" if family == "ordinary" else family
        slow_mode = "joint" if family == "ordinary" else "frozen"
        model, initialization_seconds = self.call(family, "model_initialization", lambda: self.models.make_model(
            self.memory.Config(mode, key_dim=8), seed=1001, query_period=4, slow_mode=slow_mode))
        optimizer, optimizer_initialization_seconds = self.call(
            family, "optimizer_initialization", lambda: self.core.construct_optimizer(model))

        def stage(name, start):
            self.check()
            require(self.receipt["pending"] is not None, "batch operation owns stage")
            self.receipt["pending"].update(stage=name, chunk_start=start)
            self.event({"event": "stage", "family": family, "operation": name, "chunk_start": start})

        def measured_batch():
            initial = {name: value.detach().clone() for name, value in model.named_parameters()}
            result = self.core.batch_update(model, optimizer, data, list(range(6)), check=self.check, stage=stage)
            require(result["forward_chunks"] == 8 and result["forward_rows"] == 1536
                    and result["nonquery_rows"] == 1152 and result["prior_rows"] == 378
                    and result["optimizer_updates"] == result["optimizer_step"] == 1
                    and result["backward_chunks"] == 8 and result["frozen_optimizer_state_entries"] == 0,
                    "one complete differentiated eight-chunk synthetic batch")
            effective = model.effective_named_parameters()
            require(result["effective_parameter_names"] == [name for name, _ in effective]
                    and result["effective_parameter_count"] == (6112 if family == "ordinary" else 224),
                    "only intended effective optimizer parameters")
            effective_ids = {id(parameter) for _, parameter in effective}
            changed = []
            for name, parameter in model.named_parameters():
                final = parameter.detach().clone()
                require(bool(torch.isfinite(final).all()), "finite validated final parameter copy")
                if id(parameter) in effective_ids:
                    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                            and float(optimizer.state[parameter]["step"]) == 1., "one finite effective Adam state")
                else:
                    require(parameter.grad is None and parameter not in optimizer.state
                            and torch.equal(final, initial[name]), "unchanged frozen slow parameter")
                if not torch.equal(final, initial[name]):
                    changed.append(name)
            require(bool(changed) and all(name in dict(effective) for name in changed),
                    "synthetic update changes an effective parameter only")
            return result, changed, model.parameter_metadata()

        (work, changed, parameters), seconds = self.call(family, "batch_update_and_validation", measured_batch)
        require(seconds > 0, "positive complete batch measurement")
        self.receipt["optimizer_updates"] += 1
        self.receipt["completed_families"].append(family)
        self.receipt["completed_family_count"] += 1
        return {"family": family, "batch_seconds": seconds, "forward_chunks": 8, "work": work,
                "model_initialization_seconds": initialization_seconds,
                "optimizer_initialization_seconds": optimizer_initialization_seconds,
                "changed_parameter_names": changed, "parameters": parameters,
                "timing_scope": "full batch including validated tensor copies, materialization, forward/backward, "
                                "Adam, final checks and stage-journal IO; outer attempt/return IO and setup separate"}

    def body(self):
        tick = self.clock.now_ns()
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "CPU1 Torch runtime")
        from openjev.research import otto_query_memory as memory
        from openjev.research import otto_query_memory_data as data
        from openjev.research import otto_query_memory_model as models
        from openjev.research import otto_query_memory_training as core

        self.np, self.torch, self.memory, self.data, self.models, self.core = np, torch, memory, data, models, core
        write(self.out / "runtime.json", {**runtime_record(), "torch": torch.__version__, "numpy": np.__version__,
            "environment": {name: os.environ[name] for name in THREADS}, "device": "cpu", "dtype": "float32",
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "numerical_import_seconds": (self.clock.now_ns() - tick) / 1e9})
        data, setup_seconds = self.call("shared", "synthetic_materialization", self.synthetic)
        rows = [self.one_family(family, data) for family in FAMILIES]
        require(self.receipt["completed_families"] == list(FAMILIES)
                and self.receipt["completed_family_count"] == self.receipt["optimizer_updates"] == 5
                and self.receipt["pending"] is None and self.receipt["pending_emission"] is None,
                "all five synthetic families completed without pending work")
        summary = capacity_summary(rows, self.plan["planned_chunks"])
        summary["synthetic_materialization_seconds"] = setup_seconds
        write(self.out / "summary.json", summary)
        return summary

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("original capacity supervisor stopped worker")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.admit()
            summary = self.body()
            for name, expected in self.plan["sources"].items():
                self.check()
                require(descriptor(name)["sha256"] == expected, "unchanged capacity source")
            for record in self.plan["inputs"].values():
                self.check()
                require(descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")},
                        "unchanged capacity input")
            metadata = self.plan["collection_metadata"]
            require(descriptor(metadata["path"]) == {key: metadata[key] for key in ("sha256", "bytes")}
                    and runtime_record() == self.plan["runtime"], "unchanged length metadata and runtime")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"],
                    "unchanged capacity plan/launch")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact five capacity payloads")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, admitted=summary["admitted"], files=files,
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "admitted": summary["admitted"],
                              "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, admitted=False,
                                error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication:  # noqa: BLE001 - retain the original failure
                error.add_note(f"Failure publication: {publication!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    for role in sorted(ROLES):
        plan.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
        plan.add_argument("--" + role.replace("_", "-") + "-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and Path(sys.executable).absolute() == ROOT / INTERPRETER,
            "canonical Torch interpreter and repository")
    require(args.output.is_absolute(), "exclusive absolute capacity output")
    if args.mode == "plan":
        freeze(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), "absolute original worker bindings")
        Run(args).execute()


if __name__ == "__main__":
    main()
