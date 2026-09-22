"""Frozen autonomous query allocation, original native TF and NumPy gates only.

No training, teacher annotations, replacement cases, warmup or checkpoint selection.
Metadata admission precedes every numerical import. Buffered journals stay inside
an explicitly pending episode until their shared fsync and episode acknowledgement.
"""
from __future__ import annotations

import argparse
import collections
import gzip
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
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-query-gate-evaluation-v1"
SELF = "scripts/evaluate_otto_query_gate.py"
TEST = "tests/test_evaluate_otto_query_gate.py"
PROTOCOL = "research/otto-query-gate-evaluation-protocol.md"
MODEL = "src/openjev/research/otto_query_gate_models.py"
QUALIFIER = "scripts/qualify_otto_query_gate_native.py"
QUALIFIER_PIN = "0ef5368a55dab615129602576300c3231d94fd28f7543d31354d9805ea8a5049"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
INTERPRETER = ".venv-otto-released-native/bin/python"
SEEDS, KINDS = (40101, 40102, 40103), ("gru32", "mlp190")
LEARNED = tuple(f"{kind}@{seed}" for seed in SEEDS for kind in KINDS)
ARMS = LEARNED + ("neural", "analytic")
FIRST = {"lambda3": 17100001, "lambda4": 17200001, "lambda5": 17300001}
HORIZON, CASES, EPISODES = 2188, 24, 576
LIMITS = {"native_seconds": 3600, "rss_bytes": 4 * 1024**3, "output_bytes": 6 * 1024**3}
CALL_CAPS = {"tensorflow_construction": 1, "tensorflow_build": 1, "tensorflow_load": 1,
             "tensorflow_value": 504 * HORIZON, "native_reset": EPISODES,
             "native_step": EPISODES * HORIZON, "actor_construction": EPISODES,
             "backend_binding": 432, "actor_choose": EPISODES * HORIZON,
             "actor_update": EPISODES * HORIZON, "gate_restore": 6, "numpy_gate": 432 * HORIZON}
CONFIGURATION = {"arms": list(ARMS), "first_seeds": FIRST, "cases_per_setting": CASES,
                 "horizon": HORIZON, "episodes": EPISODES, "query_threshold": .05,
                 "rotation": "global case index modulo eight", "blocks": 8,
                 "Ngrid": 53, "Nhits": 4, "R_dt": 2., "norm_Poisson": "Euclidean",
                 "primary": ["lambda3", "lambda4"], "transfer": "lambda5",
                 "reported_criteria": 62, "required_criteria": 38, "training_updates": 0}
ROLES = {f"{stage}_{part}" for stage in ("native", "collection", "training")
         for part in ("plan", "receipt", "terminal")} | {"seed_review", "kernel_lambda5", "engineering"}
JOURNALS = {"work.jsonl", "weights.jsonl", "forwards.jsonl", "gate-operations.jsonl",
            "gate-decisions.jsonl", "transitions.jsonl"}
PAYLOADS = {f"{name}.gz" for name in JOURNALS} | {
    "started.json", "runtime.json", "setup.json", "deployment.json", "episodes.jsonl", "summary.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular input")
    return path


def descriptor(path):
    path = regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(regular(path).read_text())


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


def cohort():
    result = []
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(CASES):
            offset = (ri * CASES + case) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                result.append({"regime": regime, "seed": first + case, "case": case,
                               "initial_hit": 1 + case % 3, "block": case // 3, "arm": arm,
                               "episode_id": f"eval:{regime}:{first + case}:{arm}"})
    return result


def closed_stage(paths, inputs, stage):
    plan, worker, terminal = (read(paths[f"{stage}_{p}"]) for p in ("plan", "receipt", "terminal"))
    require(worker["status"] == "completed" and terminal["status"] == "completed"
            and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is True and terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["errors"] == [] and terminal["error"] is None
            and terminal["clock_error"] is None, f"closed original {stage} stage")
    command = terminal["command"]
    launch_path = regular(command[command.index("--supervision") + 1])
    launch = read(launch_path)
    require(descriptor(launch_path)["sha256"] == worker["supervision_sha256"]
            and worker["plan_sha256"] == inputs[f"{stage}_plan"]["sha256"], f"{stage} plan/launch joins")
    for key in ("command", "cwd", "pid", "pgid", "parent_pid", "started_ns", "deadline_ns",
                "cap_seconds", "clock_backend", "clock_source_sha256", "watchdog_sha256"):
        require(launch[key] == terminal[key], f"{stage} parent identity {key}")
    require(Path(command[command.index("--plan") + 1]) == paths[f"{stage}_plan"]
            and command[command.index("--plan-sha256") + 1] == inputs[f"{stage}_plan"]["sha256"]
            and Path(command[command.index("--output") + 1]) == paths[f"{stage}_receipt"].parent,
            f"{stage} original invocation")
    require(launch["started_ns"] <= worker["started_ns"] < worker["finished_ns"] <= terminal["finished_ns"]
            <= launch["deadline_ns"] and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9,
            f"{stage} bounded physical times")
    metadata_key = "roles" if stage == "training" else "inputs"
    require(worker["sources"] == plan["sources"] and worker[metadata_key] == plan[metadata_key]
            and worker["pending"] == [], f"{stage} exact completed provenance")
    if stage == "training":
        for name, record in plan["inputs"].items():
            require(descriptor(ROOT / name) == record, "unchanged training admission input")
    directory = paths[f"{stage}_receipt"].parent
    require({p.name for p in directory.iterdir()} == set(worker["files"]) | {"receipt.json"}, f"{stage} closure")
    for name, record in worker["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == record, f"{stage} payload {name}")
    return plan, worker, terminal


def learning_admission(paths, stages):
    """Only completed saved metadata is admitted; no training code is imported."""
    collection = read(paths["collection_receipt"].parent / "summary.json")
    training = read(paths["training_receipt"].parent / "summary.json")
    require(collection["complete"] is True and stages["collection"][1]["completed_episodes"] == 60
            and len(collection["episodes"]) == 60,
            "all sixty full collection paths required")
    require(all(1 <= r["rows"] <= HORIZON and (r["found"] or r["rows"] == HORIZON)
                and r["censored"] == (not r["found"]) and r["gate_progress"]["pending_action"] is None
                and not r["gate_progress"]["pending_operations"] and not r["gate_progress"]["failed"]
                for r in collection["episodes"]), "complete collection tails")
    require(sum(r["rows"] for r in collection["episodes"]) == collection["rows"]
            and [r["episode_index"] for r in collection["episodes"]] == list(range(60)), "complete ordered TRAIN rows")
    require([{k: row[k] for k in identity} for row, identity in
             zip(collection["episodes"], stages["collection"][0]["cohort"], strict=True)]
            == stages["collection"][0]["cohort"], "unchanged sixty TRAIN case identities")
    trained = stages["training"][1]
    require(trained["qualified"] is True and trained["completed_fits"] == 6
            and trained["all_train_parity_passed"] is True, "all six fits and deployment parity completed")
    fits = training["fits"]
    require([r["fit_id"] for r in fits] == list(LEARNED), "all six fixed final fits")
    checkpoints = {}
    for fit in fits:
        require(fit["epochs"] == 80 and fit["parity"]["passed"] is True and fit["parity"]["episodes"] == 60
                and fit["parity"]["identical_query_decisions"] is True
                and fit["parity"]["rows"] == collection["rows"]
                and fit["parity"]["absolute_tolerance"] == 2e-5
                and fit["parity"]["max_logit_error"] <= 2e-5 and fit["parity"]["max_state_error"] <= 2e-5,
                "every-row final Torch/NumPy qualification")
        record = fit["final_checkpoint"]
        path = paths["training_receipt"].parent / record["path"]
        require(Path(record["path"]).name == record["path"]
                and descriptor(path) == {k: record[k] for k in ("sha256", "bytes")}, "final exported gate bytes")
        checkpoints[fit["fit_id"]] = {"path": str(path.relative_to(ROOT)), **descriptor(path)}
    require(stages["training"][0]["roles"]["collection_receipt"] == {
                "path": str(paths["collection_receipt"]), **descriptor(paths["collection_receipt"])},
            "training consumed this complete collection")
    return {"collection": collection, "training": training, "checkpoints": checkpoints,
            "collection_worker_seconds": stages["collection"][1]["wall_seconds"],
            "collection_parent_seconds": stages["collection"][2]["wall_seconds"],
            "training_worker_seconds": stages["training"][1]["wall_seconds"],
            "training_parent_seconds": stages["training"][2]["wall_seconds"]}


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact evaluation metadata roles")
    paths = {}
    for role, record in inputs.items():
        require(set(record) == {"path", "sha256", "bytes"}, "input descriptor schema")
        paths[role] = regular(record["path"])
        require(descriptor(paths[role]) == {k: record[k] for k in ("sha256", "bytes")}, f"input {role}")
    require(descriptor(ROOT / QUALIFIER)["sha256"] == QUALIFIER_PIN, "qualified native integration source")
    qualifier = load(ROOT / QUALIFIER, "_query_eval_qualified_native")
    native_plan, native_paths, reference = qualifier.authenticate(types.SimpleNamespace(
        plan=paths["native_plan"], plan_sha256=inputs["native_plan"]["sha256"]))
    stages = {stage: closed_stage(paths, inputs, stage) for stage in ("native", "collection", "training")}
    require(stages["native"][1]["qualified"] is True and stages["native"][1]["completed_episodes"] == 18,
            "completed native query integration prerequisite")
    learning = learning_admission(paths, stages)
    sources = dict(native_plan["sources"])
    for stage in ("collection", "training"):
        for name, pin in stages[stage][0]["sources"].items():
            require(name not in sources or sources[name] == pin, "consistent inherited source identities")
            sources[name] = pin
    for name in (SELF, TEST, MODEL, PROTOCOL):
        pin = descriptor(ROOT / name)["sha256"]
        require(name not in sources or sources[name] == pin, "new admission cannot replace an inherited source pin")
        sources[name] = pin
    for name, pin in sources.items():
        require(descriptor(ROOT / name)["sha256"] == pin, f"inherited source {name}")
    engineering = read(paths["engineering"])
    require(engineering["status"] == "passed" and engineering["source_before"] == engineering["source_after"]
            and engineering["source_before"][SELF] == sources[SELF] and engineering["source_before"][TEST] == sources[TEST]
            and all(x["returncode"] == 0 for x in engineering["commands"]), "qualified current evaluator metrics/lifecycle")
    for name, record in engineering["files"].items():
        require(Path(name).name == name and descriptor(paths["engineering"].parent / name) == record, "engineering log")
    seeds = read(paths["seed_review"])
    require(seeds["version"] == "otto-query-gate-learning-seeds-v1"
            and seeds["status"] == "reserved_before_collection"
            and seeds["evaluation_seeds"] == [first + case for first in FIRST.values() for case in range(CASES)]
            and seeds["reserved_seeds"] == [*range(18100001, 18100007), *range(18200001, 18200007)]
            and seeds["fit_seeds"] == list(SEEDS) and seeds["hits"] == [] and len(seeds["files"]) == 232,
            "exact disjoint evaluation seed reservation")
    return sources, native_plan["runtime"], native_paths, reference, learning


def freeze(args):
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role))
        require(descriptor(path)["sha256"] == getattr(args, f"{role}_sha256"), f"external input {role}")
        inputs[role] = {"path": str(path.relative_to(ROOT)), **descriptor(path)}
    sources, runtime, native_paths, _, learning = authenticate_inputs(inputs)
    plan = {"version": VERSION, "status": "frozen_before_autonomous_evaluation", "sources": sources,
            "inputs": inputs, "runtime": runtime, "configuration": CONFIGURATION, "limits": LIMITS,
            "call_caps": CALL_CAPS, "payloads": sorted(PAYLOADS), "cohort": cohort(),
            "native_inputs": {k: {"path": str(v.relative_to(ROOT)), **descriptor(v)} for k, v in native_paths.items()},
            "checkpoints": learning["checkpoints"]}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "plan": descriptor(args.output)}), flush=True)


class Ledger:
    """Original ForwardRecorder interface with episode-durable compressed journals."""

    def __init__(self, run):
        self.run, self.context, self.sequence = run, {"phase": "setup"}, 0
        self.pending, self.pending_emission, self.streams = [], None, {}
        self.io_seconds = 0.
        self.last_seconds, self.last_instrumented, self.last_io = {}, {}, {}
        self.calls = {k: {"attempted": 0, "returned": 0, "seconds": 0.} for k in CALL_CAPS}

    def emit(self, name, value):
        require(name in JOURNALS, "declared bounded journal")
        self.pending_emission = {"file": name, "context": dict(self.context),
                                 "event": value.get("event"), "call_id": value.get("call_id")}
        tick = time.perf_counter()
        try:
            if name not in self.streams:
                raw = (self.run.out / f"{name}.gz").open("xb")
                self.streams[name] = (raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0))
            encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            require(len(encoded) <= 64 * 1024 and self.streams[name][1].write(encoded) == len(encoded), "complete bounded event")
            self.pending_emission = None
        finally:
            self.io_seconds += time.perf_counter() - tick

    def call(self, channel, function):
        self.run.check()
        require(channel in CALL_CAPS and self.calls[channel]["attempted"] < CALL_CAPS[channel], "fixed operation cap")
        self.sequence += 1
        event = {"call_id": self.sequence, "channel": channel, "context": dict(self.context),
                 "parent_call_id": self.pending[-1]["call_id"] if self.pending else None}
        self.pending.append(event)
        self.calls[channel]["attempted"] += 1
        self.emit("work.jsonl", {"event": "attempt", **event})
        tick, io = time.perf_counter(), self.io_seconds
        result = function()
        raw, excluded = time.perf_counter() - tick, self.io_seconds - io
        elapsed = raw - excluded
        require(elapsed >= 0 and self.pending[-1] is event, "nested operation timing/order")
        self.emit("work.jsonl", {"event": "return", **event, "seconds": elapsed,
                               "instrumented_seconds": raw, "excluded_io_seconds": excluded})
        self.calls[channel]["returned"] += 1
        self.calls[channel]["seconds"] += elapsed
        self.last_seconds[channel], self.last_instrumented[channel], self.last_io[channel] = elapsed, raw, excluded
        self.pending.pop()
        self.run.check()
        return result

    def flush(self):
        tick = time.perf_counter()
        try:
            for raw, stream in self.streams.values():
                stream.flush()
                raw.flush()
                os.fsync(raw.fileno())
        finally:
            self.io_seconds += time.perf_counter() - tick

    def close(self, *, suppress=False):
        errors = []
        for name, (raw, stream) in self.streams.items():
            for handle in (stream, raw):
                try:
                    if not handle.closed:
                        if handle is raw:
                            handle.flush()
                            os.fsync(handle.fileno())
                        handle.close()
                except BaseException as error:  # noqa: BLE001 - preserve every cleanup failure and original work error
                    errors.append({"file": name, "error": repr(error)})
        if errors:
            self.run.receipt.setdefault("cleanup_errors", []).extend(errors)
            if not suppress:
                raise OSError(f"journal closure failed: {errors}")


def setup_allocation(setup, arm, gate_load_seconds, gate_module_seconds, extra_common_seconds):
    common = (setup["common_import_seconds"] + setup["shared_evaluator_setup_seconds"] + extra_common_seconds) / EPISODES
    model = setup["model_setup_seconds"] / 504 if arm != "analytic" else 0.
    gate = gate_load_seconds.get(arm, 0.) / 72 + (gate_module_seconds / 432 if arm in LEARNED else 0.)
    return {"common_setup_allocation_seconds": common, "tf_setup_allocation_seconds": model,
            "gate_setup_allocation_seconds": gate, "setup_allocation_seconds": common + model + gate}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.ledger = Ledger(self)
        self.rows, self.pending_episode, self.active_actor = [], None, None
        self.receipt = {"version": VERSION, "status": "started", "training_updates": 0,
                        "scope": "one complete frozen autonomous evaluation; no annotation or training"}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original shared evaluation deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "evaluation RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir()) <= LIMITS["output_bytes"], "evaluation output cap")

    def bind(self):
        require(descriptor(ROOT / CLOCK)["sha256"] == CLOCK_PIN, "qualified clock")
        self.clock = load(ROOT / CLOCK, "_query_eval_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch appears")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                and Path(self.launch["cwd"]) == Path.cwd() == ROOT and self.launch["cap_seconds"] == 3600
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 3600 * 10**9,
                "actual original bounded parent")
        self.check()
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external plan pin")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_autonomous_evaluation"
                and self.plan["configuration"] == CONFIGURATION and self.plan["limits"] == LIMITS
                and self.plan["call_caps"] == CALL_CAPS and self.plan["cohort"] == cohort()
                and set(self.plan["payloads"]) == PAYLOADS, "fixed complete evaluation allocation")
        sources, runtime, self.paths, self.reference, self.learning = authenticate_inputs(self.plan["inputs"])
        require(sources == self.plan["sources"] and runtime == self.plan["runtime"]
                and self.learning["checkpoints"] == self.plan["checkpoints"]
                and {k: {"path": str(v.relative_to(ROOT)), **descriptor(v)} for k, v in self.paths.items()}
                == self.plan["native_inputs"], "complete fresh admission matches frozen plan")
        require(self.launch["watchdog_sha256"] == sources["scripts/supervise_dialogue_observation_v2.py"]
                and self.launch["clock_source_sha256"] == CLOCK_PIN, "actual parent source")
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=descriptor(self.args.supervision)["sha256"],
                            sources=sources, inputs=self.plan["inputs"], limits=LIMITS)
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch,
                                         "request": {k: str(v) for k, v in vars(self.args).items()}})

    def setup(self):
        tick = time.perf_counter()
        runtime = self.reference.Run.setup(self)
        self.original_setup_wall = time.perf_counter() - tick
        tick = time.perf_counter()
        from openjev.research.otto_restricted_policy import RestrictedPolicyActor

        extra_common = time.perf_counter() - tick
        tick = time.perf_counter()
        from openjev.research.otto_query_gate import FEATURE_NAMES, QueryGateActor
        from openjev.research.otto_query_gate_models import FrozenGate
        from openjev.research.otto_reference_control import _SpaceAwarePolicy

        self.gate_module_seconds = time.perf_counter() - tick
        require("torch" not in sys.modules, "native deployment never imports Torch")
        self.gate_class, self.analytic_policy, self.restricted = QueryGateActor, _SpaceAwarePolicy, RestrictedPolicyActor
        self.feature_names = FEATURE_NAMES
        self.kernels = {"lambda3": runtime.kernels["base"], "lambda4": runtime.kernels["shift"]}
        self.mixtures = {"lambda3": runtime.weights["base"], "lambda4": runtime.weights["shift"]}
        tick = time.perf_counter()
        with runtime.np.load(ROOT / self.plan["inputs"]["kernel_lambda5"]["path"], allow_pickle=False) as archive:
            require(set(archive.files) == {"likelihood", "initial_hit_weights"}, "transfer public kernel")
            kernel, weights = archive["likelihood"], archive["initial_hit_weights"]
        require(kernel.shape == (4, 107, 107) and kernel.dtype == runtime.np.float64
                and runtime.np.isfinite(kernel).all() and weights.shape == (4,) and weights[0] == 0
                and runtime.np.isfinite(weights).all() and (weights[1:] > 0).all(), "transfer kernel/mixture geometry")
        self.kernels["lambda5"] = kernel
        self.mixtures["lambda5"] = {h: float(weights[h]) for h in (1, 2, 3)}
        self.extra_common_seconds = extra_common + time.perf_counter() - tick
        self.gates, self.gate_load_seconds = {}, {}
        for arm, record in self.plan["checkpoints"].items():
            self.ledger.context = {"phase": "gate_restore", "arm": arm}

            def restore(record=record):
                with runtime.np.load(ROOT / record["path"], allow_pickle=False) as archive:
                    return FrozenGate({name: archive[name] for name in archive.files})

            gate = self.ledger.call("gate_restore", restore)
            kind, seed = arm.split("@")
            require(gate.kind == kind and gate.seed == int(seed) and gate.state_size == (32 if kind == "gru32" else 0),
                    "six frozen gate identities")
            self.gates[arm], self.gate_load_seconds[arm] = gate, self.ledger.last_seconds["gate_restore"]
        train = self.learning["training"]
        exclusive = {fit["fit_id"]: fit["fit_seconds"] + fit["parity_seconds"] for fit in train["fits"]}
        shared = self.learning["training_parent_seconds"] - math.fsum(exclusive.values())
        require(shared >= 0, "nonoverlapping training intervals fit inside paid parent")
        amortization = {arm: {"collection_allocation_seconds": self.learning["collection_parent_seconds"] / 6,
                              "exclusive_fit_and_parity_seconds": exclusive[arm],
                              "training_common_allocation_seconds": shared / 6,
                              "paid_training_seconds": self.learning["collection_parent_seconds"] / 6 + exclusive[arm] + shared / 6}
                        for arm in LEARNED}
        for record in amortization.values():
            record["training_amortized_over_72_evaluations_seconds"] = record["paid_training_seconds"] / 72
        write(self.out / "deployment.json", {"checkpoints": self.plan["checkpoints"], "gate_load_seconds": self.gate_load_seconds,
              "gate_module_seconds": self.gate_module_seconds, "extra_common_seconds": self.extra_common_seconds,
              "original_setup_wall_seconds": self.original_setup_wall, "query_threshold": .05,
              "allocation": "common original import/evaluator setup and transfer kernel /576; original model setup /504 "
                            "only neural plus six gates; each fresh gate load /72 and gate module /432; "
                            "historical setup.json /192 is not used", "warmup_forwards": 0,
              "training_costs": {k: v for k, v in self.learning.items() if k.endswith("seconds")},
              "training_allocations": amortization,
              "training_family_allocations": {kind: math.fsum(amortization[f"{kind}@{seed}"]["paid_training_seconds"] for seed in SEEDS)
                                               for kind in KINDS},
              "training_cost_scope": "Shared collection parent time /6 plus each exclusive fit/parity interval and remaining "
              "training parent time /6. Allocations conserve physical parent totals once; /72 is explicit amortization, "
              "outside primary online controller criteria."})
        return runtime

    def episode(self, identity):
        runtime, np = self.runtime, self.runtime.np
        arm, regime = identity["arm"], identity["regime"]
        self.pending_episode = dict(identity)
        self.ledger.context = {"phase": "native_reset", **identity}
        env = self.ledger.call("native_reset", lambda: runtime.public.seeded_environment(runtime.source, identity["seed"],
            {"Ndim": 2, "Ngrid": 53, "Nhits": 4, "lambda_over_dx": float(regime[-1]),
             "R_dt": 2., "norm_Poisson": "Euclidean"}, initial_hit=identity["initial_hit"]))
        reset_seconds = self.ledger.last_seconds["native_reset"]
        require(env.p_Poisson.tobytes() == self.kernels[regime].tobytes(), "every reset supplied public kernel")
        current = self.reference.packet(runtime.public.observation(env, 0))
        recorder = self.reference.ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)
        policies, ages, gate_record = {}, collections.Counter(), {}
        last_query = None

        def decide(features, state):
            age = current["step"] if last_query is None else current["step"] - last_query
            require(features[self.feature_names.index("query_age/2188")] == np.float32(age / HORIZON), "actual query age")
            reply = self.ledger.call("numpy_gate", lambda: self.gates[arm].step(features, state))
            query, next_state, logit, probability = reply
            require(type(query) is bool and math.isfinite(logit) and math.isfinite(probability)
                    and 0 <= probability <= 1 and query == (probability >= .05), "fixed deployed threshold")
            gate_record.update(queried=query, logit=logit, probability=probability, query_age=age)
            self.ledger.emit("gate-decisions.jsonl", {**identity, "step": current["step"], **gate_record,
                "features": features.tolist(), "state_before": state.tolist(), "state_after": next_state.tolist()})
            ages[age] += 1
            return query, next_state

        self.ledger.context = {"phase": "actor_setup", **identity}

        def construct():
            if arm == "analytic":
                return runtime.analytic(current, self.kernels[regime], allow_stay=False)
            if arm == "neural":
                return self.restricted(current, self.kernels[regime], recorder, runtime.policy, sym_avg=True)
            return self.gate_class(current, self.kernels[regime], float(regime[-1]),
                lambda view: policies["analytic"]._value_policy()[1],
                lambda view: policies["neural"]._value_policy()[1], decide,
                state_size=self.gates[arm].state_size, check=self.check,
                emit=lambda event: self.ledger.emit("gate-operations.jsonl", {**identity, **event}))

        actor = self.ledger.call("actor_construction", construct)
        self.active_actor = actor
        init = self.ledger.last_seconds["actor_construction"]
        if arm in LEARNED:
            def bind_backends():
                policies["analytic"] = self.analytic_policy(env=actor.view, model=None, sym_avg=False, allow_stay=False)
                policies["neural"] = runtime.policy(env=actor.view, model=recorder, sym_avg=True)
            self.ledger.call("backend_binding", bind_backends)
            init += self.ledger.last_seconds["backend_binding"]
        witness = self.reference.belief_witness(actor, env, current, np)
        self.ledger.emit("transitions.jsonl", {"kind": "reset", **identity, "public": current,
            "posterior_after": witness, "source_evaluation_only": env.source.tolist()})
        chooses = updates = environment = raw_choose = excluded_choose = tf_seconds = 0.
        queries = 0
        for step in range(1, HORIZON + 1):
            self.ledger.context = {"phase": "decision", **identity, "step": step}
            before_tf = self.ledger.calls["tensorflow_value"]["returned"]
            action, costs = self.ledger.call("actor_choose", actor.choose)
            choice, raw, excluded = (self.ledger.last_seconds["actor_choose"], self.ledger.last_instrumented["actor_choose"],
                                     self.ledger.last_io["actor_choose"])
            queried = actor.last_choice["queried"] if arm in LEARNED else arm == "neural"
            actual_tf = self.ledger.calls["tensorflow_value"]["returned"] - before_tf
            require(actual_tf == int(queried), "one neural forward iff query, no annotations")
            require(action in current["valid_actions"] and costs.shape == (4,)
                    and costs.dtype == (np.float32 if queried else np.float64), "endpoint action/scores")
            allowed = current["valid_actions"]
            selected = costs[list(allowed)]
            require(np.isfinite(selected).all()
                    and allowed[int(np.flatnonzero(np.abs(selected - selected.min()) < 1e-10)[0])] == action,
                    "unchanged first eligible near-minimum rule")
            if queried:
                queries += 1
                last_query = current["step"]
                tf_seconds += self.ledger.last_seconds["tensorflow_value"]
            result = self.ledger.call("native_step", lambda action=action: env.step(action, quiet=True))
            native_seconds = self.ledger.last_seconds["native_step"]
            after = self.reference.packet(runtime.public.observation(env, step))
            require((int(result[0]), bool(result[2])) == (after["hit"], after["done"]), "native public event")
            self.ledger.call("actor_update", lambda action=action, after=after: actor.update(action, after))
            update = self.ledger.last_seconds["actor_update"]
            next_witness = self.reference.belief_witness(actor, env, after, np)
            require(after["position"] != current["position"], "no blocked moves")
            self.ledger.emit("transitions.jsonl", {"kind": "step", **identity, "step": step, "action": action,
                "queried": queried, "costs": [float(x) if np.isfinite(x) else None for x in costs],
                "cost_dtype": str(costs.dtype), "allowed_actions": list(allowed), "public": after,
                "posterior_before": witness, "posterior_after": next_witness, "choose_seconds": choice,
                "choose_instrumented_seconds": raw, "choose_excluded_io_seconds": excluded,
                "update_seconds": update, "environment_seconds": native_seconds, "native_p_end": float(result[1]),
                "model_forward_calls": actual_tf})
            chooses += choice
            raw_choose += raw
            excluded_choose += excluded
            updates += update
            environment += native_seconds
            current, witness = after, next_witness
            if current["done"]:
                break
        allocation = setup_allocation(runtime.setup, arm, self.gate_load_seconds, self.gate_module_seconds, self.extra_common_seconds)
        if arm in LEARNED:
            require(not actor.progress["failed"] and not actor.progress["pending_operations"]
                    and actor.progress["pending_action"] is None, "complete gate lifecycle")
        draws = [{k: row[k] for k in ("channel", "index", "uniform", "selected_index", "cdf_mass")} for row in env.draw_log]
        require([r["index"] for r in draws if r["channel"] == "source"] == [0]
                and [r["index"] for r in draws if r["channel"] == "hit"] == list(range(step - int(current["done"]))),
                "all paired categorical draws including terminal no-hit path")
        return {**identity, "steps": step, "found": current["done"], "queries": queries, "updates": step,
                "blocked_steps": 0, "init_seconds": init, "choose_seconds": chooses, "update_seconds": updates,
                **allocation, "controller_seconds": init + chooses + updates + allocation["setup_allocation_seconds"],
                "choose_instrumented_seconds": raw_choose, "choose_excluded_io_seconds": excluded_choose,
                "model_forward_seconds": tf_seconds, "environment_seconds": environment, "reset_seconds": reset_seconds,
                "source_evaluation_only": env.source.tolist(), "draws_evaluation_only": draws,
                "final_public": current, "final_update_assimilated": True,
                "gate_progress": actor.progress if arm in LEARNED else None,
                "query_age_histogram": dict(sorted(ages.items())), "maximum_query_age": max(ages, default=None)}

    def body(self):
        self.runtime = self.setup()
        self.ledger.flush()
        paired = {}
        tick = time.perf_counter()
        with (self.out / "episodes.jsonl").open("x") as episodes:
            for identity in cohort():
                row = self.episode(identity)
                case = row["regime"], row["case"]
                draws = {(x["channel"], x["index"]): x["uniform"] for x in row["draws_evaluation_only"]}
                require(len(draws) == len(row["draws_evaluation_only"]), "unique paired draw identities")
                if case in paired:
                    source, previous = paired[case]
                    require(source == row["source_evaluation_only"]
                            and all(previous[k] == draws[k] for k in previous.keys() & draws.keys()), "paired source/uniform streams")
                    previous.update(draws)
                else:
                    paired[case] = row["source_evaluation_only"], draws
                self.ledger.flush()
                episodes.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                episodes.flush()
                os.fsync(episodes.fileno())
                self.rows.append(row)
                self.pending_episode, self.active_actor = None, None
                self.check()
                if len(self.rows) % 8 == 0:
                    print(json.dumps({"phase": "evaluation", "episodes": len(self.rows), "total": EPISODES}), flush=True)
        self.evaluation_seconds = time.perf_counter() - tick
        require(not self.ledger.pending and all(x["attempted"] == x["returned"] for x in self.ledger.calls.values()),
                "all operations closed")
        require(self.ledger.calls["native_reset"]["returned"] == EPISODES
                and self.ledger.calls["native_step"]["returned"] == sum(x["steps"] for x in self.rows)
                and self.ledger.calls["tensorflow_value"]["returned"] == sum(x["queries"] for x in self.rows), "exact operation totals")
        result = aggregate(self.rows, self.mixtures)
        result.update(calls=self.ledger.calls, physical_evaluation_seconds=self.evaluation_seconds,
                      deployment=read(self.out / "deployment.json"),
                      cost_scope="Controller includes actor/filter, analytic scoring, features, gate, queries and allocated fresh "
                      "setup. Measured journal I/O excluded; nested timers overlap. Native environment and saved-data "
                      "checks are evaluator cost. Whole worker/parent times retain setup, instrumentation and closure.")
        write(self.out / "summary.json", result)

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("original evaluation terminated")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.bind()
            self.body()
            self.ledger.close()
            for name, pin in self.plan["sources"].items():
                self.check()
                require(descriptor(ROOT / name)["sha256"] == pin, "unchanged frozen source")
            for record in [*self.plan["inputs"].values(), *self.plan["native_inputs"].values(), *self.plan["checkpoints"].values()]:
                self.check()
                require(descriptor(ROOT / record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged input")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact completed output closure")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", files=files, calls=self.ledger.calls, completed_episodes=len(self.rows),
                pending=self.ledger.pending, pending_episode=self.pending_episode, started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished - self.start) / 1e9, requires_successful_original_supervisor=True,
                journal_scope="Returned calls acknowledge encoded events. Each complete episode acknowledges all journals "
                "only after flush/fsync; an interrupted episode remains explicitly pending.")
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                calls=self.ledger.calls, pending=self.ledger.pending, pending_emission=self.ledger.pending_emission,
                pending_episode=self.pending_episode, completed_episodes=len(self.rows),
                active_gate=getattr(self.active_actor, "progress", None))
            try:
                self.ledger.close(suppress=True)
            except BaseException as secondary:  # noqa: BLE001 - never skip failed receipt after cleanup failure
                self.receipt.setdefault("cleanup_errors", []).append({"error": repr(secondary)})
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original error after failed publication
                error.add_note(f"Failure publication: {secondary!r}")
            raise


def aggregate(rows, mixtures):
    identity = ("episode_id", "regime", "seed", "case", "initial_hit", "block", "arm")
    require([{k: row[k] for k in identity} for row in rows] == cohort(), "exact complete 576-case order")
    metrics = ("found", "steps", "queries", "init_seconds", "choose_seconds", "update_seconds",
               "setup_allocation_seconds", "controller_seconds", "environment_seconds")
    for row in rows:
        require(type(row["steps"]) is int and 1 <= row["steps"] <= HORIZON and type(row["found"]) is bool
                and (row["found"] or row["steps"] == HORIZON) and row["updates"] == row["steps"]
                and row["blocked_steps"] == 0 and all(math.isfinite(row[k]) and row[k] >= 0 for k in metrics),
                "complete finite outcome and cost")
        require(abs(row["controller_seconds"] - math.fsum(row[k] for k in
                    ("init_seconds", "choose_seconds", "update_seconds", "setup_allocation_seconds"))) <= 1e-9,
                "complete controller cost sum")
    groups = {k: [] for k in ("gru_absolute", "mlp_absolute", "usefulness", "recurrence", "reference")}
    regimes = {}

    def condition(group, name, value, threshold, relation):
        groups[group].append({"name": name, "value": value, "threshold": threshold, "relation": relation,
                              "passes": bool(value >= threshold if relation == ">=" else value <= threshold)})

    for regime in FIRST:
        weights = {h: float(mixtures[regime].get(h, mixtures[regime].get(str(h)))) for h in (1, 2, 3)}
        require(all(v > 0 for v in weights.values()) and abs(math.fsum(weights.values()) - 1) <= 1e-12, "fixed positive-hit mixture")
        local = [row for row in rows if row["regime"] == regime]

        def weighted(subset, metric, weights=weights):
            return math.fsum(weights[h] * math.fsum(float(r[metric]) for r in subset if r["initial_hit"] == h)
                             / sum(r["initial_hit"] == h for r in subset) for h in (1, 2, 3))

        means = {a: {m: weighted([r for r in local if r["arm"] == a], m) for m in metrics} for a in ARMS}
        families = {k: {m: math.fsum(means[f"{k}@{s}"][m] for s in SEEDS) / 3 for m in metrics} for k in KINDS}
        blocks = [{a: {m: weighted([r for r in local if r["arm"] == a and r["block"] == b], m) for m in metrics}
                   for a in ARMS} for b in range(8)]
        if regime != "lambda5":
            for control in ("neural", "analytic"):
                condition("reference", f"{regime}.{control}.success", means[control]["found"], .95, ">=")
            for arm in LEARNED:
                group = "gru_absolute" if arm.startswith("gru32@") else "mlp_absolute"
                own = means[arm]
                condition(group, f"{regime}.{arm}.success", own["found"], .95, ">=")
                for control in ("neural", "analytic"):
                    condition(group, f"{regime}.{arm}.moves_vs_{control}", own["steps"], 1.05 * means[control]["steps"], "<=")
                condition(group, f"{regime}.{arm}.cost_vs_neural", own["controller_seconds"], .5 * means["neural"]["controller_seconds"], "<=")
            gru, mlp, analytic = families["gru32"], families["mlp190"], means["analytic"]
            condition("usefulness", f"{regime}.success_vs_analytic", gru["found"], analytic["found"], ">=")
            condition("usefulness", f"{regime}.moves_vs_analytic", gru["steps"], .95 * analytic["steps"], "<=")
            condition("recurrence", f"{regime}.success_vs_mlp", gru["found"], mlp["found"], ">=")
            condition("recurrence", f"{regime}.moves_vs_mlp", gru["steps"], mlp["steps"], "<=")
            condition("recurrence", f"{regime}.cost_vs_mlp", gru["controller_seconds"], .9 * mlp["controller_seconds"], "<=")
        ages = {}
        for arm in LEARNED:
            histogram = collections.Counter()
            for row in local:
                if row["arm"] == arm:
                    histogram.update({int(k): int(v) for k, v in row["query_age_histogram"].items()})
            ages[arm] = {"histogram": dict(sorted(histogram.items())), "maximum": max(histogram, default=None)}
        regimes[regime] = {"weights": weights, "means": means, "family_means": families, "blocks": blocks,
            "strata": {h: {a: {m: math.fsum(float(r[m]) for r in local if r["arm"] == a and r["initial_hit"] == h) / 8
                                for m in metrics} for a in ARMS} for h in (1, 2, 3)},
            "paired_family_block_differences": [{m: math.fsum(b[f"mlp190@{s}"][m] - b[f"gru32@{s}"][m] for s in SEEDS) / 3
                                                 for m in ("found", "steps", "controller_seconds", "queries")} for b in blocks],
            "query_ages": ages, "raw_counts": {a: {"found": sum(r["found"] for r in local if r["arm"] == a), "episodes": 24} for a in ARMS}}
    require({k: len(v) for k, v in groups.items()} == {"gru_absolute": 24, "mlp_absolute": 24, "usefulness": 4,
            "recurrence": 6, "reference": 4}, "all 62 prespecified comparisons")
    required = [row for group, records in groups.items() if group != "mlp_absolute" for row in records]
    return {"version": VERSION, "episodes": EPISODES, "paired_cases": 72, "regimes": regimes, "criteria": groups,
            "required_conditions": 38, "required_passed": sum(x["passes"] for x in required),
            "reported_conditions": 62, "reported_passed": sum(x["passes"] for rows in groups.values() for x in rows),
            "pilot_continuation": all(x["passes"] for x in required),
            "scope": "Primary continuation requires GRU24+usefulness4+recurrence6+reference4; MLP24 diagnostic. "
                     "Transfer is descriptive; no architecture novelty is established by this screen."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    plan = commands.add_parser("plan")
    for role in sorted(ROLES):
        plan.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
        plan.add_argument(f"--{role.replace('_', '-')}-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / INTERPRETER, "qualified original native interpreter")
    require(args.output.is_absolute(), "absolute exclusive output")
    if args.mode == "plan":
        freeze(args)
    else:
        Run(args).execute()


if __name__ == "__main__":
    main()
