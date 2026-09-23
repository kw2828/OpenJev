"""Fixed sparse-query VALID calibration and fresh paired autonomous comparison.

Metadata admission precedes numerical imports. Copied compressed journals use
an explicit episode durability boundary: no completed episode is acknowledged
until all of its journals have been flushed/fsynced. No fitting or annotations.
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
VERSION = "otto-sparse-query-study-v1"
SELF, TEST = "scripts/study_otto_sparse_query.py", "tests/test_study_otto_sparse_query.py"
PROTOCOL = "research/otto-sparse-query-protocol.md"
GATE, METRICS = "src/openjev/research/otto_sparse_query.py", "src/openjev/research/otto_sparse_metrics.py"
COLLECTOR = "scripts/collect_otto_query_gate.py"
COLLECTOR_PIN = "5e2b06fc0ff5b888122e733f12b20c881deab7f56b94908ed5a283f7a48c76df"
COLLECTION_PLAN_PIN = "f16f23f82d9eb271ac3c1a15df0c567f0a782b169abcdd4711de8fb9a7700be6"
TRANSFER_PLAN_PIN = "e58bbde3732ea404843bfab763b1690a998926526a509e1ee49fb9634ca905d4"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
INTERPRETER = ".venv-otto-released-native/bin/python"
ARMS = ("analytic", "neural", "period2", "random_pair", "entropy")
SPARSE = ARMS[2:]
VALID_FIRST = {"lambda3": 20100001, "lambda4": 20200001}
FIRST = {"lambda3": 20300001, "lambda4": 20400001, "lambda5": 20500001}
HORIZON, VALID_EPISODES, EVAL_EPISODES, EPISODES = 2188, 24, 360, 384
LIMITS = {"native_seconds": 1800, "rss_bytes": 4 * 1024**3, "output_bytes": 6 * 1024**3}
CALL_CAPS = {"tensorflow_construction": 1, "tensorflow_build": 1, "tensorflow_load": 1,
    "tensorflow_value": 72 * HORIZON + 240 * (HORIZON // 2), "native_reset": EPISODES, "native_step": EPISODES * HORIZON,
    "actor_construction": EPISODES, "backend_binding": 240, "actor_choose": EPISODES * HORIZON,
    "actor_update": EPISODES * HORIZON, "sparse_gate": 240 * HORIZON, "threshold_selection": 1}
CONFIGURATION = {"arms": list(ARMS), "validation_first_seeds": VALID_FIRST, "evaluation_first_seeds": FIRST,
    "validation_cases_per_setting": 12, "evaluation_cases_per_setting": 24, "horizon": HORIZON,
    "validation_episodes": VALID_EPISODES, "evaluation_episodes": EVAL_EPISODES,
    "validation_arm": "period2", "rotation": "global EVAL case index modulo five",
    "initial_hit": "1 + case % 3", "blocks": 8, "Ngrid": 53, "Nhits": 4, "R_dt": 2.,
    "norm_Poisson": "Euclidean", "primary_settings": ["lambda3", "lambda4"], "transfer": "lambda5",
    "threshold": "global episode-balanced lower weighted median of float32 feature19; entropy >= threshold",
    "quota": "every sparse prefix Q_t <= ceil(t/2)", "primary_arm": "period2", "required_criteria": 16,
    "training_updates": 0, "annotations": 0}
ROLES = {"collection_plan", "transfer_plan", "kernel_lambda5", "seed_review", "engineering"}
NEW_COMPONENTS = {SELF, TEST, GATE, "tests/test_otto_sparse_query.py", METRICS,
    "tests/test_otto_sparse_metrics.py", "scripts/audit_otto_sparse_query.py", "tests/test_audit_otto_sparse_query.py"}
NEW_SOURCES = NEW_COMPONENTS | {PROTOCOL}
JOURNALS = {"work.jsonl", "weights.jsonl", "forwards.jsonl", "gate-operations.jsonl",
            "gate-decisions.jsonl", "transitions.jsonl", "validation-decisions.jsonl"}
PAYLOADS = {f"{name}.gz" for name in JOURNALS} | {"started.json", "runtime.json", "setup.json",
    "deployment.json", "episode-boundaries.jsonl", "episodes.jsonl", "threshold.json", "costs.json", "summary.json"}

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


def cohort(stage=None):
    result = []
    for regime, first in VALID_FIRST.items():
        for case in range(12):
            result.append({"stage": "valid", "episode_index": len(result), "regime": regime,
                "seed": first + case, "case": case, "initial_hit": 1 + case % 3, "block": case // 3,
                "arm": "period2", "episode_id": f"valid:{regime}:{first + case}:period2"})
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            offset = (ri * 24 + case) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                result.append({"stage": "eval", "episode_index": len(result), "regime": regime,
                    "seed": first + case, "case": case, "initial_hit": 1 + case % 3, "block": case // 3,
                    "arm": arm, "episode_id": f"eval:{regime}:{first + case}:{arm}"})
    return result if stage is None else [r for r in result if r["stage"] == stage]


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact sparse-query roles")
    paths = {}
    for role, record in inputs.items():
        require(set(record) == {"path", "sha256", "bytes"}, "input descriptor")
        paths[role] = regular(record["path"])
        require(descriptor(paths[role]) == {k: record[k] for k in ("sha256", "bytes")}, "frozen input " + role)
    require(inputs["collection_plan"]["sha256"] == COLLECTION_PLAN_PIN
            and descriptor(ROOT / COLLECTOR)["sha256"] == COLLECTOR_PIN,
            "original qualified collection authentication source and plan")
    collector = load(ROOT / COLLECTOR, "_sparse_qualified_collection")
    prior, native_paths, reference, native = collector.authenticate(types.SimpleNamespace(
        plan=paths["collection_plan"], plan_sha256=COLLECTION_PLAN_PIN))
    require(inputs["transfer_plan"]["sha256"] == TRANSFER_PLAN_PIN, "original transfer kernel lineage")
    transfer = read(paths["transfer_plan"])
    require(transfer["inputs"]["kernel_lambda5"] == inputs["kernel_lambda5"], "exact inherited transfer kernel")
    sources = dict(prior["sources"])
    for name in NEW_SOURCES:
        pin = descriptor(ROOT / name)["sha256"]
        require(name not in sources or sources[name] == pin, "new source cannot replace inherited source")
        sources[name] = pin
    engineering = read(paths["engineering"])
    require(engineering["status"] == "passed" and engineering["sources_before"] == engineering["sources_after"]
            and set(engineering["sources_after"]) == NEW_COMPONENTS and engineering["results"]
            and all(type(r["exit_code"]) is int and r["exit_code"] == 0 for r in engineering["results"]),
            "all current fabricated component qualifications")
    for name, pin in engineering["sources_after"].items():
        require(sources[name] == pin, "current qualified component")
    directory = paths["engineering"].parent
    require({p.name for p in directory.iterdir()} == set(engineering["files"]) | {"receipt.json"}, "engineering closure")
    for name, d in engineering["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == d, "engineering log")
    seeds = read(paths["seed_review"])
    require(seeds["status"] == "reserved_before_run" and seeds["hits"] == [] and seeds["files"]
            and seeds["validation_seeds"] == [first + case for first in VALID_FIRST.values() for case in range(12)]
            and seeds["evaluation_seeds"] == [first + case for first in FIRST.values() for case in range(24)],
            "scoped exact fresh seed reservation")
    return sources, prior["runtime"], native_paths, reference, native


def freeze(args):
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role)); d = descriptor(path)
        require(d["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        inputs[role] = {"path": str(path.relative_to(ROOT)), **d}
    sources, runtime, native_paths, _, _ = authenticate_inputs(inputs)
    plan = {"version": VERSION, "status": "frozen_before_validation", "configuration": CONFIGURATION,
        "sources": sources, "inputs": inputs, "runtime": runtime, "limits": LIMITS, "call_caps": CALL_CAPS,
        "payloads": sorted(PAYLOADS), "cohort": cohort(),
        "native_inputs": {k: {"path": str(p.relative_to(ROOT)), **descriptor(p)} for k, p in native_paths.items()}}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "plan": descriptor(args.output)}), flush=True)


def setup_allocation(setup, arm, gate_module_seconds, extra_common_seconds):
    common = (setup["common_import_seconds"] + setup["shared_evaluator_setup_seconds"] + extra_common_seconds) / 384
    model = setup["model_setup_seconds"] / 312 if arm != "analytic" else 0.
    gate = gate_module_seconds / 240 if arm in SPARSE else 0.
    return {"common_setup_allocation_seconds": common, "tf_setup_allocation_seconds": model,
            "gate_setup_allocation_seconds": gate, "setup_allocation_seconds": common + model + gate}


def paired_identity(row, paired):
    key = row["regime"], row["case"]
    draws = {(d["channel"], d["index"]): d["uniform"] for d in row["draws_evaluation_only"]}
    require(len(draws) == len(row["draws_evaluation_only"]), "unique paired draw identities")
    if key in paired:
        source, previous = paired[key]
        require(source == row["source_evaluation_only"] and all(previous[k] == draws[k]
                for k in previous.keys() & draws.keys()), "paired original source and uniform streams")
        previous.update(draws)
    else:
        paired[key] = row["source_evaluation_only"], draws

class Ledger:
    """Copied qualified compressed ledger; returns are provisional until episode fsync."""

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



class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.ledger = Ledger(self)
        self.rows, self.validation_decisions = [], []
        self.pending_episode = self.active_actor = None
        self.threshold = self.threshold_descriptor = None
        self.receipt = {"version": VERSION, "status": "started", "training_updates": 0, "annotations": 0,
            "scope": "Fixed VALID threshold then complete fresh EVAL; no learned gate or fitting."}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original shared sparse-query deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "sparse-query RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                <= LIMITS["output_bytes"] - 1024**2, "output cap with failure reserve")

    def append_durable(self, name, value):
        require(name in ("episode-boundaries.jsonl", "episodes.jsonl"), "declared durable boundary")
        self.check()
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        require(len(encoded) < 2 * 1024**2, "bounded episode record")
        with (self.out / name).open("ab") as stream:
            require(stream.write(encoded) == len(encoded), "complete boundary write")
            stream.flush()
            os.fsync(stream.fileno())

    def bind(self):
        require(descriptor(ROOT / CLOCK)["sha256"] == CLOCK_PIN, "clock pin before import")
        self.clock = load(ROOT / CLOCK, "_sparse_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and command[:3] == [str(ROOT / INTERPRETER), str(ROOT / SELF), "run"]
            and self.launch["pid"] == os.getpid() and self.launch["pgid"] == os.getpgrp()
            and self.launch["parent_pid"] == os.getppid() and self.launch["pid"] != self.launch["parent_pid"]
            and Path(self.launch["cwd"]) == Path.cwd() == ROOT and self.launch["cap_seconds"] == 1800
            and self.launch["clock_backend"] == self.clock.backend
            and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
            and self.launch["deadline_ns"] == self.launch["started_ns"] + 1800 * 10**9,
            "canonical original bounded parent")
        self.check()
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external plan pin")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_validation"
            and self.plan["configuration"] == CONFIGURATION and self.plan["limits"] == LIMITS
            and self.plan["call_caps"] == CALL_CAPS and self.plan["cohort"] == cohort()
            and set(self.plan["payloads"]) == PAYLOADS, "fixed complete allocation")
        sources, runtime, self.paths, self.reference, self.native = authenticate_inputs(self.plan["inputs"])
        require(sources == self.plan["sources"] and runtime == self.plan["runtime"]
            and {k: {"path": str(p.relative_to(ROOT)), **descriptor(p)} for k, p in self.paths.items()}
            == self.plan["native_inputs"], "unchanged native/source/runtime admission")
        require(self.launch["watchdog_sha256"] == sources["scripts/supervise_dialogue_observation_v2.py"]
            and self.launch["clock_source_sha256"] == CLOCK_PIN, "original parent source")
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=descriptor(self.args.supervision)["sha256"],
            sources=sources, inputs=self.plan["inputs"], native_inputs=self.plan["native_inputs"], limits=LIMITS)
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
        from openjev.research import otto_sparse_metrics
        from openjev.research.otto_query_gate import FEATURE_NAMES, QueryGateActor
        from openjev.research.otto_reference_control import _SpaceAwarePolicy
        from openjev.research.otto_sparse_query import SparseQueryGate
        self.gate_module_seconds = time.perf_counter() - tick
        require("torch" not in sys.modules, "no Torch or trained gate")
        self.gate_class, self.analytic_policy, self.restricted = QueryGateActor, _SpaceAwarePolicy, RestrictedPolicyActor
        self.sparse_class, self.metrics, self.feature_names = SparseQueryGate, otto_sparse_metrics, FEATURE_NAMES
        self.kernels = {"lambda3": runtime.kernels["base"], "lambda4": runtime.kernels["shift"]}
        self.mixtures = {"lambda3": runtime.weights["base"], "lambda4": runtime.weights["shift"]}
        tick = time.perf_counter()
        with runtime.np.load(ROOT / self.plan["inputs"]["kernel_lambda5"]["path"], allow_pickle=False) as archive:
            require(set(archive.files) == {"likelihood", "initial_hit_weights"}, "transfer public kernel")
            kernel, weights = archive["likelihood"], archive["initial_hit_weights"]
        require(kernel.shape == (4, 107, 107) and kernel.dtype == runtime.np.float64
            and runtime.np.isfinite(kernel).all() and (kernel >= 0).all() and (kernel[:, 53, 53] == 0).all()
            and weights.shape == (4,) and weights[0] == 0 and runtime.np.isfinite(weights).all()
            and (weights[1:] > 0).all() and abs(float(weights.sum()) - 1) <= 1e-10, "transfer kernel and mixture")
        self.kernels["lambda5"] = kernel
        self.mixtures["lambda5"] = {h: float(weights[h]) for h in (1, 2, 3)}
        self.extra_common_seconds = extra_common + time.perf_counter() - tick
        self.deploy = {"gate_module_seconds": self.gate_module_seconds, "extra_common_seconds": self.extra_common_seconds,
            "original_setup_wall_seconds": self.original_setup_wall, "warmup_forwards": 0,
            "allocation": "common /384; original TF model /312 neural-capable paths; sparse modules /240 gated paths; "
                          "24 VALID setup shares plus full calibration wall charged only to entropy over72 EVAL paths",
            "mixtures": self.mixtures, "feature_names": list(self.feature_names),
            "reference": "Unchanged original analytic and restricted TensorFlow endpoints; no gate callback for controls."}
        write(self.out / "deployment.json", self.deploy)
        return runtime

    def episode(self, identity):
        runtime, np = self.runtime, self.runtime.np
        arm, regime = identity["arm"], identity["regime"]
        self.pending_episode = dict(identity)
        self.append_durable("episode-boundaries.jsonl", {"event": "attempt", **identity})
        self.ledger.context = {"phase": "native_reset", **identity}
        env = self.ledger.call("native_reset", lambda: runtime.public.seeded_environment(runtime.source, identity["seed"],
            {"Ndim": 2, "Ngrid": 53, "Nhits": 4, "lambda_over_dx": float(regime[-1]),
             "R_dt": 2., "norm_Poisson": "Euclidean"}, initial_hit=identity["initial_hit"]))
        reset_seconds = self.ledger.last_seconds["native_reset"]
        require(env.p_Poisson.tobytes() == self.kernels[regime].tobytes(), "reset equals authenticated public kernel")
        current = self.reference.packet(runtime.public.observation(env, 0))
        recorder = self.reference.ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)
        policies, ages, gate_record = {}, collections.Counter(), {}
        last_query, queries, quota_valid = None, 0, True
        schedule = None

        def decide(features, state):
            age = current["step"] if last_query is None else current["step"] - last_query
            require(features[self.feature_names.index("query_age/2188")] == np.float32(age / HORIZON), "actual query age")
            query, next_state, record = self.ledger.call("sparse_gate", lambda: schedule.decision(features, state))
            require(type(query) is bool and record["query"] is query and record["step"] == current["step"]
                and record["queries_before"] == queries and record["queries_after"] == queries + int(query)
                and record["queries_after"] <= (current["step"] + 2) // 2, "causal sparse quota each prefix")
            gate_record.clear()
            gate_record.update(record)
            self.ledger.emit("gate-decisions.jsonl", {**identity, **record, "query_age": age,
                "features": features.tolist(), "state_before": state.tolist(), "state_after": next_state.tolist()})
            if identity["stage"] == "valid":
                saved = {**identity, "step": current["step"], "entropy": float(features[19]), "queried": query}
                self.ledger.emit("validation-decisions.jsonl", saved)
                self.validation_decisions.append(saved)
            ages[age] += 1
            return query, next_state

        self.ledger.context = {"phase": "actor_setup", **identity}

        def construct():
            nonlocal schedule
            if arm == "analytic":
                return runtime.analytic(current, self.kernels[regime], allow_stay=False)
            if arm == "neural":
                return self.restricted(current, self.kernels[regime], recorder, runtime.policy, sym_avg=True)
            schedule = self.sparse_class(arm, episode_seed=identity["seed"],
                entropy_threshold=self.threshold["value"] if arm == "entropy" else None)
            return self.gate_class(current, self.kernels[regime], float(regime[-1]),
                lambda view: policies["analytic"]._value_policy()[1],
                lambda view: policies["neural"]._value_policy()[1], decide, state_size=2,
                check=self.check, emit=lambda event: self.ledger.emit("gate-operations.jsonl", {**identity, **event}))

        actor = self.ledger.call("actor_construction", construct)
        self.active_actor = actor
        init = self.ledger.last_seconds["actor_construction"]
        if arm in SPARSE:
            def bind_backends():
                policies["analytic"] = self.analytic_policy(env=actor.view, model=None, sym_avg=False, allow_stay=False)
                policies["neural"] = runtime.policy(env=actor.view, model=recorder, sym_avg=True)
            self.ledger.call("backend_binding", bind_backends)
            init += self.ledger.last_seconds["backend_binding"]
        witness = self.reference.belief_witness(actor, env, current, np)
        self.ledger.emit("transitions.jsonl", {"kind": "reset", **identity, "public": current,
            "posterior_after": witness, "source_evaluation_only": env.source.tolist()})
        chooses = updates = environment = raw_choose = excluded_choose = tf_seconds = 0.
        for step in range(1, HORIZON + 1):
            self.ledger.context = {"phase": "decision", **identity, "step": step}
            before_tf = self.ledger.calls["tensorflow_value"]["returned"]
            action, costs = self.ledger.call("actor_choose", actor.choose)
            choice, raw, excluded = (self.ledger.last_seconds["actor_choose"], self.ledger.last_instrumented["actor_choose"],
                                     self.ledger.last_io["actor_choose"])
            queried = actor.last_choice["queried"] if arm in SPARSE else arm == "neural"
            actual_tf = self.ledger.calls["tensorflow_value"]["returned"] - before_tf
            require(actual_tf == int(queried), "one original neural forward iff queried; no annotation")
            require(action in current["valid_actions"] and costs.shape == (4,)
                and costs.dtype == (np.float32 if queried else np.float64), "unchanged endpoint scores")
            allowed = current["valid_actions"]
            selected = costs[list(allowed)]
            require(np.isfinite(selected).all()
                and allowed[int(np.flatnonzero(np.abs(selected - selected.min()) < 1e-10)[0])] == action,
                "unchanged dtype-preserving first near-minimum action")
            if queried:
                queries += 1
                last_query = current["step"]
                tf_seconds += self.ledger.last_seconds["tensorflow_value"]
            if arm in SPARSE:
                quota_valid = quota_valid and queries <= (step + 1) // 2
                require(quota_valid and queries == gate_record["queries_after"], "every executed query obeys causal quota")
            result = self.ledger.call("native_step", lambda action=action: env.step(action, quiet=True))
            native_seconds = self.ledger.last_seconds["native_step"]
            after = self.reference.packet(runtime.public.observation(env, step))
            require((int(result[0]), bool(result[2])) == (after["hit"], after["done"]), "original public observation")
            self.ledger.call("actor_update", lambda action=action, after=after: actor.update(action, after))
            update = self.ledger.last_seconds["actor_update"]
            next_witness = self.reference.belief_witness(actor, env, after, np)
            require(after["position"] != current["position"], "no blocked movement")
            self.ledger.emit("transitions.jsonl", {"kind": "step", **identity, "step": step, "action": action,
                "queried": queried, "costs": [float(x) if np.isfinite(x) else None for x in costs],
                "cost_dtype": str(costs.dtype), "allowed_actions": list(allowed), "public": after,
                "posterior_before": witness, "posterior_after": next_witness, "choose_seconds": choice,
                "choose_instrumented_seconds": raw, "choose_excluded_io_seconds": excluded,
                "update_seconds": update, "environment_seconds": native_seconds, "native_p_end": float(result[1]),
                "model_forward_calls": actual_tf, "query_quota_valid": quota_valid if arm in SPARSE else None})
            chooses += choice; raw_choose += raw; excluded_choose += excluded
            updates += update; environment += native_seconds
            current, witness = after, next_witness
            if current["done"]:
                break
        allocation = setup_allocation(runtime.setup, arm, self.gate_module_seconds, self.extra_common_seconds)
        if arm in SPARSE:
            require(not actor.progress["failed"] and not actor.progress["pending_operations"]
                and actor.progress["pending_action"] is None, "complete sparse actor lifecycle")
        draws = [{k: row[k] for k in ("channel", "index", "uniform", "selected_index", "cdf_mass")} for row in env.draw_log]
        require([r["index"] for r in draws if r["channel"] == "source"] == [0]
            and [r["index"] for r in draws if r["channel"] == "hit"] == list(range(step - int(current["done"]))),
            "all categorical draws with terminal no-hit semantics")
        return {**identity, "steps": step, "found": current["done"], "queries": queries, "updates": step,
            "blocked_steps": 0, "init_seconds": init, "choose_seconds": chooses, "update_seconds": updates,
            **allocation, "controller_seconds": init + chooses + updates + allocation["setup_allocation_seconds"],
            "choose_instrumented_seconds": raw_choose, "choose_excluded_io_seconds": excluded_choose,
            "model_forward_seconds": tf_seconds, "environment_seconds": environment, "reset_seconds": reset_seconds,
            "source_evaluation_only": env.source.tolist(), "draws_evaluation_only": draws,
            "final_public": current, "final_update_assimilated": True,
            "gate_progress": actor.progress if arm in SPARSE else None, "query_quota_valid": quota_valid,
            "query_age_histogram": dict(sorted(ages.items())), "maximum_query_age": max(ages, default=None)}

    def complete_episode(self, identity, paired):
        row = self.episode(identity)
        if identity["stage"] == "eval":
            paired_identity(row, paired)
        self.ledger.flush()
        self.append_durable("episodes.jsonl", row)
        self.append_durable("episode-boundaries.jsonl", {"event": "return", **identity, "steps": row["steps"]})
        self.rows.append(row)
        self.pending_episode, self.active_actor = None, None
        self.check()
        return row

    def calibrate(self):
        tick = time.perf_counter()
        for identity in cohort("valid"):
            self.complete_episode(identity, {})
        self.validation_wall_seconds = time.perf_counter() - tick
        require(len(self.rows) == 24 and [r["episode_index"] for r in self.rows] == list(range(24))
            and self.pending_episode is None and not self.ledger.pending, "all VALID closed before threshold")
        tick = time.perf_counter()
        self.ledger.context = {"phase": "threshold_from_all_valid"}
        self.threshold = self.ledger.call("threshold_selection", lambda: self.metrics.select_threshold(
            self.validation_decisions, self.rows))
        require(type(self.threshold["value"]) in (int, float) and math.isfinite(self.threshold["value"])
            and self.threshold["episodes"] == 24 and self.threshold["decisions"] == len(self.validation_decisions),
            "one complete fixed VALID threshold")
        self.ledger.flush()
        write(self.out / "threshold.json", self.threshold)
        self.threshold_descriptor = descriptor(self.out / "threshold.json")
        self.append_durable("episode-boundaries.jsonl", {"event": "threshold_published",
            "threshold": self.threshold_descriptor, "completed_validation_episodes": 24,
            "completed_evaluation_episodes": 0})
        self.threshold_wall_seconds = time.perf_counter() - tick
        self.validation_setup_seconds = math.fsum(r["setup_allocation_seconds"] for r in self.rows)
        self.calibration_paid_seconds = self.validation_wall_seconds + self.threshold_wall_seconds + self.validation_setup_seconds
        self.check()

    def body(self):
        self.runtime = self.setup()
        self.ledger.flush()
        self.calibrate()
        require(self.threshold_descriptor == descriptor(self.out / "threshold.json")
            and len(self.rows) == VALID_EPISODES, "durable threshold barrier before EVAL")
        tick = time.perf_counter(); paired = {}
        for identity in cohort("eval"):
            self.complete_episode(identity, paired)
            if (len(self.rows) - VALID_EPISODES) % 5 == 0:
                print(json.dumps({"phase": "eval", "episodes": len(self.rows) - VALID_EPISODES, "total": EVAL_EPISODES}), flush=True)
        self.evaluation_wall_seconds = time.perf_counter() - tick
        require(not self.ledger.pending and self.ledger.pending_emission is None
            and all(x["attempted"] == x["returned"] for x in self.ledger.calls.values()), "complete returned operations")
        require(len(self.rows) == EPISODES and self.ledger.calls["native_reset"]["returned"] == EPISODES
            and self.ledger.calls["native_step"]["returned"] == sum(r["steps"] for r in self.rows)
            and self.ledger.calls["tensorflow_value"]["returned"] == sum(r["queries"] for r in self.rows)
            and self.ledger.calls["sparse_gate"]["returned"] == sum(r["steps"] for r in self.rows if r["arm"] in SPARSE),
            "exact complete path/query counts")
        costs = {"original_setup_wall_seconds": self.original_setup_wall, "validation_wall_seconds": self.validation_wall_seconds,
            "threshold_wall_seconds": self.threshold_wall_seconds, "validation_setup_seconds": self.validation_setup_seconds,
            "calibration_paid_seconds": self.calibration_paid_seconds, "evaluation_wall_seconds": self.evaluation_wall_seconds,
            "journal_io_seconds": self.ledger.io_seconds, "operation_seconds": self.ledger.calls,
            "scope": "Calibration is full VALID loop plus threshold construction/publication and its24 allocated setup shares. "
                     "Charge only entropy /72 EVAL paths. Online controller includes nested monitoring, feature/gate/filter "
                     "and endpoint scoring; subtract only measured serialization/I/O. Operation timers overlap. "
                     "Environment, pairing/posterior checks and logging remain paid in full physical worker/parent time."}
        result = self.metrics.aggregate(self.rows[VALID_EPISODES:], self.mixtures, self.calibration_paid_seconds)
        result.update(version=VERSION, costs=costs, threshold=self.threshold, deployment=self.deploy,
            validation_episodes=VALID_EPISODES, evaluation_episodes=EVAL_EPISODES, training_updates=0, annotations=0)
        write(self.out / "costs.json", costs)
        write(self.out / "summary.json", result)

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)
        def interrupt(_signum, _frame):
            raise InterruptedError("original sparse-query supervisor stopped worker")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.bind()
            self.body()
            self.ledger.close()
            for name, pin in self.plan["sources"].items():
                self.check(); require(descriptor(ROOT / name)["sha256"] == pin, "unchanged source")
            for record in [*self.plan["inputs"].values(), *self.plan["native_inputs"].values()]:
                self.check()
                require(descriptor(ROOT / record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged input")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            require(descriptor(self.out / "threshold.json") == self.threshold_descriptor, "threshold fixed through EVAL")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact completed payload closure")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check(); finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, calls=self.ledger.calls,
                completed_episodes=len(self.rows), validation_episodes=24, evaluation_episodes=360,
                threshold=self.threshold_descriptor, pending=self.ledger.pending, pending_emission=self.ledger.pending_emission,
                pending_episode=self.pending_episode, started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9, requires_successful_original_supervisor=True,
                journal_scope="Encoded returns are provisional within a pending episode; all journals flush/fsync before "
                              "durable episode row and completion boundary. Interrupted episodes are failures, never resumed.")
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc(),
                calls=self.ledger.calls, pending=self.ledger.pending, pending_emission=self.ledger.pending_emission,
                pending_episode=self.pending_episode, completed_episodes=len(self.rows),
                active_gate=getattr(self.active_actor, "progress", None))
            try:
                self.ledger.close(suppress=True)
            except BaseException as cleanup:  # noqa: BLE001 - preserve primary error
                self.receipt.setdefault("cleanup_errors", []).append({"error": repr(cleanup)})
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication:  # noqa: BLE001 - preserve primary error
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
    require(Path(sys.executable).absolute() == ROOT / INTERPRETER and Path.cwd() == ROOT, "original native runtime/cwd")
    require(args.output.is_absolute(), "exclusive absolute output")
    if args.mode == "plan":
        freeze(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), "absolute worker bindings")
        Run(args).execute()


if __name__ == "__main__":
    main()
