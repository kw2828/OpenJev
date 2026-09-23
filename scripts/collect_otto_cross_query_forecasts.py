"""Complete public score-forecast TRAIN/VALID collection.

Metadata admission precedes numerical imports. Copied compressed journals use
an explicit episode durability boundary: no completed episode is acknowledged
until all of its journals have been flushed/fsynced. TRAIN scores every period-four query row plus sampled windows; VALID is a census. No fitting.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
import types
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-cross-query-forecast-collection-v1"
SELF, TEST = "scripts/collect_otto_cross_query_forecasts.py", "tests/test_collect_otto_cross_query_forecasts.py"
PROTOCOL = "research/otto-cross-query-forecast-protocol.md"
COLLECTOR = "scripts/collect_otto_query_gate.py"
COLLECTOR_PIN = "5e2b06fc0ff5b888122e733f12b20c881deab7f56b94908ed5a283f7a48c76df"
COLLECTION_PLAN_PIN = "f16f23f82d9eb271ac3c1a15df0c567f0a782b169abcdd4711de8fb9a7700be6"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
INTERPRETER = ".venv-otto-released-native/bin/python"
ARMS = ("analytic", "neural", "period4_hold")
FIRST = {"train": {"lambda3": 251000001, "lambda4": 252000001},
         "valid": {"lambda3": 253000001, "lambda4": 254000001}}
CASES = {"train": 9, "valid": 6}
HORIZON, EPISODES = 2188, 90
LIMITS = {"native_seconds": 7200, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
# 18 neural TRAIN paths; 36 analytic/held TRAIN paths each require at most
# 547 query anchors plus 8*3 selected nonquery rows; 36 full-census VALID paths.
# 18*2188 + 36*(547 + 8*3) + 36*2188 = 138708 teacher/TF calls.
CALL_CAPS = {"tensorflow_construction": 1, "tensorflow_build": 1, "tensorflow_load": 1,
    "tensorflow_value": 138708, "native_reset": EPISODES, "native_step": EPISODES * HORIZON,
    "actor_construction": EPISODES, "backend_binding": EPISODES + 54, "analytic_score": EPISODES * HORIZON,
    "feature_build": EPISODES * HORIZON, "teacher_score": 138708,
    "actor_update": EPISODES * HORIZON, "annotation_public_reset": 54,
    "annotation_public_update": 54 * HORIZON}
CONFIGURATION = {"arms": list(ARMS), "first_seeds": FIRST, "cases": CASES, "horizon": HORIZON,
    "train_episodes": 54, "valid_episodes": 36, "rotation": "global case index modulo three",
    "initial_hit": "1 + case % 3", "Ngrid": 53, "Nhits": 4, "R_dt": 2., "norm_Poisson": "Euclidean",
    "period": 4, "train_sampling": "uniform min(8,ceil(length/4)) windows without replacement after each complete path",
    "selection_seed_first": 256000001, "validation_annotation": "complete census",
    "training_annotation": "union of all period-four query rows and selected windows; query anchors are not extra loss targets",
    "features": "virtual last_query=step-step%4, age=step%4, has_queried=1, including correction rows",
    "annotation": "skip action fixed before annotation; annotation never refreshes held cache",
    "training_updates": 0, "warmup_forwards": 0}
ROLES = {"collection_plan", "seed_review", "engineering"}
NEW_COMPONENTS = {SELF, TEST, "src/openjev/research/otto_cross_query_scores.py",
    "tests/test_otto_cross_query_scores.py", "src/openjev/research/otto_score_forecast_data.py",
    "tests/test_otto_score_forecast_data.py", "src/openjev/research/otto_sampled_forecast_data.py",
    "tests/test_otto_sampled_forecast_data.py"}
NEW_SOURCES = NEW_COMPONENTS | {PROTOCOL}
JOURNALS = {"work.jsonl", "weights.jsonl", "forwards.jsonl", "transitions.jsonl", "samples.jsonl", "annotations.jsonl"}
PAYLOADS = {f"{name}.gz" for name in JOURNALS} | {"started.json", "runtime.json", "setup.json",
    "deployment.json", "cohort.json", "episode-boundaries.jsonl", "episodes.jsonl", "train.npz",
    "valid.npz", "costs.json", "summary.json", "train-selection.json"}
ARRAY_KEYS = ("features", "raw_q", "legal", "actions", "correction", "episode_offsets")

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
    result, global_case = [], 0
    for split, regimes in FIRST.items():
        for regime, first in regimes.items():
            for case in range(CASES[split]):
                offset = global_case % len(ARMS)
                for arm in ARMS[offset:] + ARMS[:offset]:
                    result.append({"stage": split, "episode_index": len(result), "regime": regime,
                        "seed": first + case, "case": case, "initial_hit": 1 + case % 3,
                        "arm": arm, "episode_id": f"{split}:{regime}:{first + case}:{arm}"})
                global_case += 1
    return result if stage is None else [r for r in result if r["stage"] == stage]


def virtual_query(step):
    require(type(step) is int and 0 <= step < HORIZON, "preaction step")
    return step - step % 4


def route_scores(arm, step, allowed, analytic_action, teacher, select, cache, *, annotate=True):
    """Deployment is unchanged; optional census annotations never change held scores."""
    require(arm in ARMS and callable(teacher) and callable(select), "fixed controller and scorers")
    correction = virtual_query(step) == step
    deployed = arm == "neural" or (arm == "period4_hold" and correction)
    before = None if cache is None else cache.tobytes()
    action = analytic_action if arm == "analytic" else None
    if arm == "period4_hold" and not correction:
        require(cache is not None, "held controller requires earlier correction")
        action = select(cache, allowed)
    raw = teacher(deployed) if deployed or annotate else None
    require(cache is None or cache.tobytes() == before, "annotation cannot modify held memory")
    if deployed:
        action = select(raw, allowed)
        if arm == "period4_hold":
            cache = raw.copy()
            cache.flags.writeable = False
    require(action in allowed, "inbounds actual action")
    return int(action), raw, cache, deployed, correction


def episode_offsets(lengths):
    require(lengths and all(type(n) is int and 1 <= n <= HORIZON for n in lengths), "all complete positive episode lengths")
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    return offsets


def posterior_witness(view):
    probability = view.p_source
    return {"mass": float(probability.sum()), "sha256": hashlib.sha256(probability.tobytes()).hexdigest(),
            "exact": True}


def selected_steps(selection):
    return tuple(step for start in selection["start_offsets"]
                 for step in range(start, min(start + 4, selection["length"])))


def train_annotation_steps(selection):
    """All query anchors plus the selected loss windows, without duplicate calls.

    This does not change the uniform window sample. Unselected query rows are
    model inputs for chronological scans; they do not become new loss targets.
    """
    return tuple(sorted(set(range(0, selection["length"], 4)) | set(selected_steps(selection))))


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact forecast collection roles")
    paths = {}
    for role, record in inputs.items():
        require(set(record) == {"path", "sha256", "bytes"}, "input descriptor")
        paths[role] = regular(record["path"])
        require(descriptor(paths[role]) == {k: record[k] for k in ("sha256", "bytes")}, "frozen input " + role)
    require(inputs["collection_plan"]["sha256"] == COLLECTION_PLAN_PIN
            and descriptor(ROOT / COLLECTOR)["sha256"] == COLLECTOR_PIN,
            "original qualified collection authentication source and plan")
    collector = load(ROOT / COLLECTOR, "_forecast_qualified_collection")
    prior, native_paths, reference, native = collector.authenticate(types.SimpleNamespace(
        plan=paths["collection_plan"], plan_sha256=COLLECTION_PLAN_PIN))
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
    expected_seeds = [first + case for split, regimes in FIRST.items()
                      for first in regimes.values() for case in range(CASES[split])] + [255000001, 255000002, 255000003] + list(range(256000001, 256000055))
    require(seeds["status"] == "reserved_before_run" and seeds["hits"] == [] and seeds["files"]
            and seeds["seeds"] == expected_seeds, "scoped exact fresh seed reservation")
    return sources, prior["runtime"], native_paths, reference, native


def freeze(args):
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role)); d = descriptor(path)
        require(d["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        inputs[role] = {"path": str(path.relative_to(ROOT)), **d}
    sources, runtime, native_paths, _, _ = authenticate_inputs(inputs)
    plan = {"version": VERSION, "status": "frozen_before_collection", "configuration": CONFIGURATION,
        "sources": sources, "inputs": inputs, "runtime": runtime, "limits": LIMITS, "call_caps": CALL_CAPS,
        "payloads": sorted(PAYLOADS), "cohort": cohort(),
        "native_inputs": {k: {"path": str(p.relative_to(ROOT)), **descriptor(p)} for k, p in native_paths.items()}}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "plan": descriptor(args.output)}), flush=True)


def paired_identity(row, paired):
    key = row["stage"], row["regime"], row["case"]
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
        self.rows, self.datasets, self.selections, self.label_mask = [], {}, [], []
        self.pending_episode = self.active_actor = self.pending_action = None
        self.buffer = {key: [] for key in ARRAY_KEYS if key != "episode_offsets"}
        self.receipt = {"version": VERSION, "status": "started", "training_updates": 0,
            "scope": "Full fixed paths; every TRAIN query anchor plus sampled windows, and census VALID. No fitting."}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original shared score-forecast deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "score-forecast RSS cap")
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
        self.clock = load(ROOT / CLOCK, "_forecast_clock").SuspendClock()
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
            and Path(self.launch["cwd"]) == Path.cwd() == ROOT and self.launch["cap_seconds"] == 7200
            and self.launch["clock_backend"] == self.clock.backend
            and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
            and self.launch["deadline_ns"] == self.launch["started_ns"] + 7200 * 10**9,
            "canonical original bounded parent")
        self.check()
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external plan pin")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_collection"
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
        from openjev.research.otto_query_gate import FEATURE_NAMES, ReadOnlyBeliefView, _analytic, _features
        from openjev.research.otto_released_policy import PublicBeliefView, _packet
        from openjev.research.otto_restricted_policy import select_inbounds_action
        from openjev.research.otto_sampled_forecast_data import select_windows
        self.feature_names, self.readonly = FEATURE_NAMES, ReadOnlyBeliefView
        self.analytic_scores, self.features, self.select = _analytic, _features, select_inbounds_action
        self.public_view, self.packet, self.sample_windows = PublicBeliefView, _packet, select_windows
        self.feature_module_seconds = time.perf_counter() - tick
        require("torch" not in sys.modules, "no learner imports or fitting")
        self.kernels = {"lambda3": runtime.kernels["base"], "lambda4": runtime.kernels["shift"]}
        self.mixtures = {"lambda3": runtime.weights["base"], "lambda4": runtime.weights["shift"]}
        write(self.out / "deployment.json", {"original_setup_wall_seconds": self.original_setup_wall,
            "feature_module_seconds": self.feature_module_seconds, "warmup_forwards": 0,
            "feature_names": list(self.feature_names), "mixtures": self.mixtures,
            "features": CONFIGURATION["features"], "teacher": "original restricted float32 four-action costs",
            "scope": "All full paths; TRAIN retains every query anchor and samples at most eight loss windows after termination. VALID annotates every state. Deployed and deferred scores are distinct. No deployment-cost claim."})
        write(self.out / "cohort.json", {"episodes": cohort(), "configuration": CONFIGURATION})
        return runtime

    def episode(self, identity):
        runtime, np = self.runtime, self.runtime.np
        arm, regime = identity["arm"], identity["regime"]
        start_row = len(self.buffer["features"])
        self.pending_episode = dict(identity)
        self.append_durable("episode-boundaries.jsonl", {"event": "attempt", **identity, "start_row": start_row})
        self.ledger.context = {"phase": "native_reset", **identity}
        env = self.ledger.call("native_reset", lambda: runtime.public.seeded_environment(runtime.source, identity["seed"],
            {"Ndim": 2, "Ngrid": 53, "Nhits": 4, "lambda_over_dx": float(regime[-1]),
             "R_dt": 2., "norm_Poisson": "Euclidean"}, initial_hit=identity["initial_hit"]))
        require(env.p_Poisson.tobytes() == self.kernels[regime].tobytes(), "reset equals authenticated kernel")
        current = self.reference.packet(runtime.public.observation(env, 0))
        self.ledger.context = {"phase": "actor_setup", **identity}
        actor = self.ledger.call("actor_construction", lambda: runtime.analytic(current, self.kernels[regime], allow_stay=False))
        self.active_actor = actor
        view = self.readonly(actor._view)
        recorder = self.reference.ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)
        teacher = self.ledger.call("backend_binding", lambda: runtime.policy(env=view, model=recorder, sym_avg=True))
        witness = self.reference.belief_witness(actor, env, current, np)
        self.ledger.emit("transitions.jsonl", {"kind": "reset", **identity, "public": current,
            "posterior_after": witness, "source_evaluation_only": env.source.tolist()})
        cache = previous = last_action = None
        history = []
        deployed = annotations = corrections = 0
        for step in range(HORIZON):
            self.ledger.context = {"phase": "preaction", **identity, "step": step}
            allowed = current["valid_actions"]
            analytic_action, analytic = self.ledger.call("analytic_score", lambda allowed=allowed: self.analytic_scores(
                actor._policy._value_policy()[1], allowed))
            features, next_statistics = self.ledger.call("feature_build", lambda step=step, current=current, analytic=analytic, last_action=last_action, previous=previous: self.features(
                view, current, float(regime[-1]), analytic, last_action, virtual_query(step), previous))
            require(features[15] == np.float32(step / HORIZON) and features[16] == np.float32((step % 4) / HORIZON)
                and features[17] == np.float32(1), "declared virtual correction features")
            public_before, belief_before = actor.public, actor.belief.tobytes()
            tf_before = self.ledger.calls["tensorflow_value"]["returned"]

            def score(is_deployed, step=step, public_before=public_before, belief_before=belief_before):
                self.ledger.context = {"phase": "teacher_score", **identity, "step": step,
                    "deployed_query": is_deployed, "annotation_only": not is_deployed}
                raw = self.ledger.call("teacher_score", lambda: teacher._value_policy()[1])
                require(raw.shape == (4,) and raw.dtype == np.float32 and np.isfinite(raw).all(), "all four raw teacher costs")
                require(actor.public == public_before and actor.belief.tobytes() == belief_before
                    and actor._pending_action is None, "functional annotation cannot change actor state")
                return raw.copy()

            def select_and_record(scores, permitted, step=step):
                action = self.select(scores, permitted)
                self.pending_action = {**identity, "step": step, "action": action}
                return action
            if arm == "analytic":
                self.pending_action = {**identity, "step": step, "action": analytic_action}
            action, raw, cache, queried, correction = route_scores(
                arm, step, allowed, analytic_action, score, select_and_record, cache,
                annotate=identity["stage"] == "valid")
            require(self.ledger.calls["tensorflow_value"]["returned"] - tf_before == int(raw is not None),
                    "one physical forward only for a deployed or census annotation request")
            deployed += int(queried); annotations += int(raw is not None and not queried); corrections += int(correction)
            legal = np.asarray([a in allowed for a in range(4)], dtype=np.bool_)
            row_index = len(self.buffer["features"])
            self.ledger.emit("samples.jsonl", {**identity, "row_index": row_index, "step": step,
                "public": current, "posterior": witness, "features": features.tolist(),
                "raw_q": None if raw is None else raw.tolist(), "label_available": raw is not None,
                "legal": legal.tolist(), "action": action, "analytic_action": analytic_action,
                "deployed_query": queried, "annotation_only": raw is not None and not queried, "correction_scheduled": correction,
                "virtual_last_query": virtual_query(step),
                "held_q": None if cache is None else cache.tolist(),
                "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "raw_q_sha256": None if raw is None else hashlib.sha256(raw.tobytes()).hexdigest()})
            self.ledger.context = {"phase": "transition", **identity, "step": step + 1}
            result = self.ledger.call("native_step", lambda action=action: env.step(action, quiet=True))
            after = self.reference.packet(runtime.public.observation(env, step + 1))
            require((int(result[0]), bool(result[2])) == (after["hit"], after["done"]), "native public observation")
            self.ledger.call("actor_update", lambda action=action, after=after: actor.update(action, after))
            next_witness = self.reference.belief_witness(actor, env, after, np)
            require(after["position"] != current["position"], "inbounds movement")
            self.ledger.emit("transitions.jsonl", {"kind": "step", **identity, "step": step + 1,
                "row_index": row_index, "action": action, "deployed_query": queried, "public": after,
                "posterior_before": witness, "posterior_after": next_witness, "native_p_end": float(result[1])})
            history.append({"public": current, "posterior": witness, "after": after,
                            "posterior_after": next_witness, "action": action})
            for key, value in zip(ARRAY_KEYS[:-1], (features.copy(), np.zeros(4, dtype=np.float32) if raw is None
                                                  else raw.copy(), legal, action, correction), strict=True):
                self.buffer[key].append(value)
            self.label_mask.append(raw is not None)
            self.pending_action = None
            current, witness, last_action, previous = after, next_witness, action, next_statistics
            if current["done"]:
                break
        steps = step + 1
        require(deployed == (steps if arm == "neural" else (steps + 3) // 4 if arm == "period4_hold" else 0)
            and (identity["stage"] == "train" or deployed + annotations == steps)
            and corrections == (steps + 3) // 4, "controller and annotation accounting")
        deferred, bindings = (0, 0)
        if identity["stage"] == "train":
            deferred, bindings = self.annotate_train(identity, history, start_row, cache)
            annotations += deferred
        draws = [{k: r[k] for k in ("channel", "index", "uniform", "selected_index", "cdf_mass")} for r in env.draw_log]
        require([r["index"] for r in draws if r["channel"] == "source"] == [0]
            and [r["index"] for r in draws if r["channel"] == "hit"] == list(range(steps - int(current["done"]))),
            "all native draw identities and final no-hit semantics")
        return {**identity, "steps": steps, "rows": steps, "start_row": start_row,
            "end_row": len(self.buffer["features"]), "found": current["done"], "censored": not current["done"],
            "deployed_queries": deployed, "annotation_only": annotations, "teacher_calls": deployed + annotations,
            "deferred_annotations": deferred, "backend_bindings": 1 + bindings,
            "unlabeled_rows": sum(not v for v in self.label_mask[start_row:start_row + steps]),
            "corrections": corrections, "updates": steps, "final_public": current, "final_update_assimilated": True,
            "source_evaluation_only": env.source.tolist(), "draws_evaluation_only": draws}

    def annotate_train(self, identity, history, start_row, held):
        """Score the query/window union after the full path, without changing actions."""
        np, runtime = self.runtime.np, self.runtime
        selection = self.sample_windows(len(history), 256000001 + identity["episode_index"])
        required = set(train_annotation_steps(selection))
        record = {"episode_id": identity["episode_id"], **selection}
        self.ledger.context = {"phase": "train_selection", **identity}
        self.ledger.emit("annotations.jsonl", {"kind": "selection", **identity, "selection": selection})
        self.ledger.flush()  # The selection is durable before any deferred teacher score.
        self.selections.append(record)
        view = self.public_view(self.kernels[identity["regime"]])
        self.ledger.context = {"phase": "annotation_reset", **identity}
        self.ledger.call("annotation_public_reset", lambda: view._reset(identity["initial_hit"]))
        require(posterior_witness(view) == history[0]["posterior"], "exact public replay reset")
        teacher, count, bindings = None, 0, 0
        held_before = None if held is None else held.tobytes()
        actor_before = self.active_actor.public, self.active_actor.belief.tobytes()
        for step, entry in enumerate(history):
            self.check()
            packet = self.packet(entry["public"], step)
            require(tuple(view.agent) == packet["position"] and not packet["done"]
                    and posterior_witness(view) == entry["posterior"], "exact chronological public replay")
            row_index = start_row + step
            if step in required and not self.label_mask[row_index]:
                if teacher is None:
                    self.ledger.context = {"phase": "annotation_binding", **identity, "step": step}
                    recorder = self.reference.ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)
                    teacher = self.ledger.call("backend_binding", lambda recorder=recorder: runtime.policy(
                        env=self.readonly(view), model=recorder, sym_avg=True))
                    bindings += 1
                self.ledger.context = {"phase": "deferred_annotation", **identity, "step": step,
                                       "deployed_query": False, "annotation_only": True}
                before = posterior_witness(view)
                raw = self.ledger.call("teacher_score", lambda teacher=teacher: teacher._value_policy()[1])
                require(raw.shape == (4,) and raw.dtype == np.float32 and np.isfinite(raw).all()
                        and posterior_witness(view) == before, "finite functional deferred annotation")
                self.ledger.emit("annotations.jsonl", {"kind": "score", **identity, "step": step,
                    "row_index": row_index, "posterior": before, "raw_q": raw.tolist(),
                    "raw_q_sha256": hashlib.sha256(raw.tobytes()).hexdigest()})
                self.buffer["raw_q"][row_index] = raw.copy()
                self.label_mask[row_index] = True
                count += 1
            after = self.packet(entry["after"], step + 1)
            successor, possible = view._move(entry["action"], view.agent)
            require(possible and tuple(successor) == after["position"], "public replay uses actual recorded action")
            self.ledger.context = {"phase": "annotation_update", **identity, "step": step + 1}
            self.ledger.call("annotation_public_update", lambda after=after: view._observe(after))
            require(posterior_witness(view) == entry["posterior_after"], "exact public replay includes final update")
        require(all(self.label_mask[start_row + step] for step in required),
                "all TRAIN query anchors and selected window labels present")
        require((None if held is None else held.tobytes()) == held_before
                and (self.active_actor.public, self.active_actor.belief.tobytes()) == actor_before,
                "deferred annotations cannot mutate deployed memory or terminal actor")
        return count, bindings

    def complete_episode(self, identity, paired):
        row = self.episode(identity)
        paired_identity(row, paired)
        self.ledger.flush()
        self.append_durable("episodes.jsonl", row)
        self.append_durable("episode-boundaries.jsonl", {"event": "return", **identity, "steps": row["steps"],
            "start_row": row["start_row"], "end_row": row["end_row"]})
        self.rows.append(row)
        self.pending_episode, self.active_actor = None, None
        self.check()

    def save_stage(self, stage):
        np = self.runtime.np
        rows = [r for r in self.rows if r["stage"] == stage]
        expected = cohort(stage)
        require([{k: r[k] for k in e} for r, e in zip(rows, expected, strict=True)] == expected, "all stage identities")
        offsets = episode_offsets([r["steps"] for r in rows])
        require([(r["start_row"], r["end_row"]) for r in rows] == list(pairwise(offsets)),
            "full chronological per-episode slices")
        arrays = {key: np.asarray(self.buffer[key], dtype=dtype) for key, dtype in zip(ARRAY_KEYS[:-1],
            (np.float32, np.float32, np.bool_, np.int64, np.bool_), strict=True)}
        arrays["episode_offsets"] = np.asarray(offsets, dtype=np.int64)
        n = offsets[-1]
        require([arrays[k].shape for k in ARRAY_KEYS] == [(n, 31), (n, 4), (n, 4), (n,), (n,), (len(rows) + 1,)],
            "exact flat stage schema")
        require(np.isfinite(arrays["features"]).all() and np.isfinite(arrays["raw_q"]).all(), "finite public dataset")
        mask = np.asarray(self.label_mask, dtype=np.bool_)
        require(mask.shape == (n,) and bool((arrays["raw_q"][~mask] == 0).all()), "explicit unscored zero placeholders")
        if stage == "train":
            require(len(self.selections) == 54 and [r["episode_id"] for r in self.selections]
                    == [r["episode_id"] for r in rows], "all TRAIN selections in episode order")
            for row, selection in zip(rows, self.selections, strict=True):
                require(all(mask[row["start_row"] + step] for step in train_annotation_steps(selection)),
                        "all TRAIN query anchors and selected labels exported")
            arrays["label_mask"] = mask
            write(self.out / "train-selection.json", self.selections)
        else:
            require(bool(mask.all()), "complete VALID teacher census")
        path = self.out / f"{stage}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(saved[k].dtype == arrays[k].dtype
                and saved[k].shape == arrays[k].shape and saved[k].tobytes() == arrays[k].tobytes() for k in arrays),
                "saved flat dataset byte equality")
        self.datasets[stage] = {"path": path.name, **descriptor(path), "episodes": len(rows), "rows": n,
            "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "bytes": v.nbytes,
                           "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}
        self.buffer = {key: [] for key in ARRAY_KEYS[:-1]}
        self.label_mask = []
        self.check()

    def body(self):
        self.runtime = self.setup()
        self.ledger.flush()
        timings, paired = {}, {}
        for stage in FIRST:
            tick = time.perf_counter()
            for identity in cohort(stage):
                self.complete_episode(identity, paired)
                if len(self.rows) % 3 == 0:
                    print(json.dumps({"phase": stage, "completed_episodes": len(self.rows), "total": EPISODES}), flush=True)
            timings[stage + "_collection_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter(); self.save_stage(stage)
            timings[stage + "_serialization_seconds"] = time.perf_counter() - tick
        n = sum(r["steps"] for r in self.rows)
        require(len(self.rows) == EPISODES and not self.ledger.pending and self.ledger.pending_emission is None
            and all(x["attempted"] == x["returned"] for x in self.ledger.calls.values()), "all episodes and calls complete")
        for channel in ("native_step", "actor_update", "analytic_score", "feature_build"):
            require(self.ledger.calls[channel]["returned"] == n, "one operation per retained state: " + channel)
        teacher_calls = sum(r["teacher_calls"] for r in self.rows)
        require(all(self.ledger.calls[k]["returned"] == teacher_calls for k in ("teacher_score", "tensorflow_value")),
                "all deployed and annotation-only teacher calls")
        require(self.ledger.calls["backend_binding"]["returned"] == sum(r["backend_bindings"] for r in self.rows)
                and self.ledger.calls["annotation_public_reset"]["returned"] == 54
                and self.ledger.calls["annotation_public_update"]["returned"]
                == sum(r["steps"] for r in self.rows if r["stage"] == "train"), "all pure TRAIN replay work")
        for channel in ("native_reset", "actor_construction"):
            require(self.ledger.calls[channel]["returned"] == EPISODES, "one operation per episode: " + channel)
        costs = {"original_setup_wall_seconds": self.original_setup_wall, "feature_module_seconds": self.feature_module_seconds,
            **timings, "journal_io_seconds": self.ledger.io_seconds, "operation_seconds": self.ledger.calls,
            "scope": "Setup, TRAIN loop, TRAIN serialization, VALID loop and VALID serialization are disjoint physical intervals. "
                     "Operation timers overlap; I/O timer is partial and overlaps those intervals. Source authentication, final "
                     "hashing and process overhead remain paid in original worker/parent time. Physical teacher annotations "
                     "are included; these are collection costs, not deployed-controller cost measurements."}
        summary = {"version": VERSION, "complete": True, "episodes": EPISODES, "rows": n,
            "datasets": self.datasets, "teacher_calls": teacher_calls, "deployed_queries": sum(r["deployed_queries"] for r in self.rows),
            "annotation_only": sum(r["annotation_only"] for r in self.rows), "corrections": sum(r["corrections"] for r in self.rows),
            "found": sum(r["found"] for r in self.rows), "censored": sum(r["censored"] for r in self.rows),
            "by_stage_arm": [{"stage": stage, "arm": arm, "episodes": len(group),
                "rows": sum(r["rows"] for r in group), "deployed_queries": sum(r["deployed_queries"] for r in group),
                "annotation_only": sum(r["annotation_only"] for r in group)} for stage in FIRST for arm in ARMS
                for group in [[r for r in self.rows if r["stage"] == stage and r["arm"] == arm]]],
            "costs": costs, "training_updates": 0, "efficacy_claim": False}
        write(self.out / "costs.json", costs)
        write(self.out / "summary.json", summary)

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)
        def interrupt(_signum, _frame):
            raise InterruptedError("original score-forecast supervisor stopped worker")
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
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact completed payload closure")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check(); finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, calls=self.ledger.calls,
                completed_episodes=len(self.rows), train_episodes=54, valid_episodes=36,
                datasets=self.datasets, pending=self.ledger.pending, pending_emission=self.ledger.pending_emission,
                pending_episode=self.pending_episode, pending_action=self.pending_action, started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9, requires_successful_original_supervisor=True,
                journal_scope="Encoded returns are provisional within a pending episode; all journals flush/fsync before "
                              "durable episode row and completion boundary. Interrupted episodes are failures, never resumed.")
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc(),
                calls=self.ledger.calls, pending=self.ledger.pending, pending_emission=self.ledger.pending_emission,
                pending_episode=self.pending_episode, pending_action=self.pending_action, completed_episodes=len(self.rows),
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
