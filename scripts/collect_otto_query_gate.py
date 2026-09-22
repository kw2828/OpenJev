"""Collect complete TRAIN paths with physical query schedules and separate annotations.

Planning is metadata-only. Original TF/model/native imports follow the frozen
input and successful integration closure. No optimizer, learned gate or EVAL.
"""
from __future__ import annotations

import argparse
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-query-gate-collection-v1"
SELF = "scripts/collect_otto_query_gate.py"
TEST = "tests/test_collect_otto_query_gate.py"
PROTOCOL = "research/otto-query-gate-collection-protocol.md"
DESIGN = "research/otto-query-gate-learning-design.md"
NATIVE = "scripts/qualify_otto_query_gate_native.py"
NATIVE_PIN = "0ef5368a55dab615129602576300c3231d94fd28f7543d31354d9805ea8a5049"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
INTERPRETER = ".venv-otto-released-native/bin/python"
SCHEDULES = ("always", "never", "period2", "period8", "initial_only")
HORIZON, EPISODES, MAX_ROWS = 2188, 60, 131280
CONFIGURATION = {
    "schedules": list(SCHEDULES), "regimes": {"base": 3., "shift": 4.},
    "first_seeds": {"base": 18100001, "shift": 18200001}, "cases_per_regime": 6,
    "initial_hit": "1 + case % 3", "rotation": "left by (regime_index * 6 + case) % 5",
    "horizon": HORIZON, "episodes": EPISODES, "maximum_rows": MAX_ROWS,
    "feature_dim": 31, "symmetry_average": True, "annotation_calls_per_row": 1,
    "label": "analytic action outside eligible float32 abs(cost-min) < 1e-10 set",
    "skipped_annotation": "after actual analytic choice, before native step; cannot mutate gate history",
    "training_updates": 0, "evaluation_episodes": 0,
}
LIMITS = {"native_seconds": 600, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3,
          "native_resets": EPISODES, "native_steps": MAX_ROWS, "tensorflow_value_calls": MAX_ROWS}
CALL_CAPS = {"tensorflow_construction": 1, "tensorflow_build": 1, "tensorflow_load": 1,
             "tensorflow_value": MAX_ROWS, "native_reset": EPISODES, "native_step": MAX_ROWS,
             "actor_construction": EPISODES, "backend_binding": EPISODES,
             "actor_choose": MAX_ROWS, "actor_update": MAX_ROWS, "annotation_score": MAX_ROWS,
             "dataset_allocate": 1, "dataset_save": 1}
ROLES = {"native_plan", "native_receipt", "native_terminal", "seed_review"}
NEW_SOURCES = {SELF, TEST, PROTOCOL, DESIGN, "research/otto-query-gate-training-protocol.md",
               "research/otto-query-gate-evaluation-protocol.md"}
ARRAY_KEYS = {"features", "labels", "neural_costs", "analytic_action", "masks", "episode_offsets",
              "scheduled_query", "neural_gap"}
PAYLOADS = {"started.json", "runtime.json", "setup.json", "work.jsonl", "weights.jsonl",
            "forwards.jsonl", "gate-operations.jsonl", "public-transitions.jsonl", "episodes.jsonl",
            "training-data.npz", "collection-costs.json", "summary.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    value = Path(value)
    path = value if value.is_absolute() else ROOT / value
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular file")
    return path


def descriptor(path):
    path = regular(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return {"sha256": result.hexdigest(), "bytes": path.stat().st_size}


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
    for regime_index, (regime, sensing) in enumerate(CONFIGURATION["regimes"].items()):
        for case in range(6):
            shift = (regime_index * 6 + case) % len(SCHEDULES)
            for schedule in SCHEDULES[shift:] + SCHEDULES[:shift]:
                seed = CONFIGURATION["first_seeds"][regime] + case
                result.append({"episode_id": f"train:{regime}:{seed}:{schedule}", "episode_index": len(result),
                               "regime": regime, "sensing_length": sensing, "case": case, "seed": seed,
                               "initial_hit": 1 + case % 3, "schedule": schedule})
    return result


def scheduled_query(schedule, step):
    require(schedule in SCHEDULES and type(step) is int and 0 <= step < HORIZON, "declared schedule and preaction step")
    return (schedule == "always" or schedule == "period2" and step % 2 == 0
            or schedule == "period8" and step % 8 == 0 or schedule == "initial_only" and step == 0)


def annotation_label(scores, allowed, analytic_action, np):
    require(isinstance(scores, np.ndarray) and scores.dtype == np.float32 and scores.shape == (4,)
            and np.isfinite(scores).all(), "four original finite float32 scores")
    require(isinstance(allowed, (list, tuple)) and all(type(a) is int and 0 <= a < 4 for a in allowed)
            and tuple(allowed) == tuple(sorted(set(allowed))) and allowed
            and type(analytic_action) is int and analytic_action in allowed, "ordered eligibility and analytic action")
    permitted = scores[list(allowed)]
    minimum = permitted.min()
    near = np.abs(permitted - minimum) < 1e-10
    # Selection can differ within the qualified near-minimum set without a label.
    gap = np.float32(scores[analytic_action] - minimum)
    require(np.isfinite(gap) and gap >= 0, "finite nonnegative original float32 neural gap")
    return not bool(near[list(allowed).index(analytic_action)]), gap


def closed(directory, files, names):
    require(set(files) == set(names) and {p.name for p in directory.iterdir()} == set(names) | {"receipt.json"},
            "exact successful closed inventory")
    for name, desc in files.items():
        require(Path(name).name == name and descriptor(directory / name) == desc, "closed payload hash")


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact collection input roles")
    paths = {}
    for role, record in inputs.items():
        require(set(record) == {"path", "sha256", "bytes"}, "input descriptor")
        paths[role] = regular(record["path"])
        require(descriptor(paths[role]) == {k: record[k] for k in ("sha256", "bytes")}, f"input {role}")
    require(descriptor(ROOT / NATIVE)["sha256"] == NATIVE_PIN, "qualified integration source before import")
    native = load(ROOT / NATIVE, "_query_collection_native")
    prior, inherited, reference = native.authenticate(types.SimpleNamespace(
        plan=paths["native_plan"], plan_sha256=inputs["native_plan"]["sha256"]))
    receipt = read(paths["native_receipt"])
    require(receipt["version"] == native.VERSION and receipt["status"] == "completed" and receipt["qualified"] is True
            and receipt["plan_sha256"] == inputs["native_plan"]["sha256"] and receipt["pending"] == []
            and receipt["sources"] == prior["sources"] and receipt["inputs"] == prior["inputs"]
            and receipt["native_inputs"] == prior["native_inputs"] and receipt["completed_episodes"] == 18,
            "successful exact native integration")
    closed(paths["native_receipt"].parent, receipt["files"], native.PAYLOADS)
    started = read(paths["native_receipt"].parent / "started.json")
    terminal = read(paths["native_terminal"])
    launch_path = regular(started["request"]["supervision"])
    launch = read(launch_path)
    require(descriptor(launch_path)["sha256"] == receipt["supervision_sha256"] and started["launch"] == launch,
            "original native launch identity")
    require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["errors"] == [] and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["cap_seconds"] == 180 and terminal["timing_available"] is True
            and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"] <= terminal["finished_ns"]
            <= terminal["deadline_ns"], "original successful bounded native parent")
    for key in ("command", "cwd", "pid", "pgid", "parent_pid", "started_ns", "deadline_ns", "cap_seconds",
                "clock_backend", "clock_source_sha256", "watchdog_sha256"):
        require(terminal[key] == launch[key], f"native parent/launch {key}")
    command = list(terminal["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command == [str(ROOT / INTERPRETER), NATIVE, "run", "--plan", str(paths["native_plan"]),
                       "--plan-sha256", inputs["native_plan"]["sha256"], "--supervision", str(launch_path),
                       "--output", str(paths["native_receipt"].parent)], "actual native command and input joins")
    seeds = read(paths["seed_review"])
    require(seeds["version"] == "otto-query-gate-learning-seeds-v1" and seeds["status"] == "reserved_before_collection"
            and seeds["reserved_seeds"] == [*range(18100001, 18100007), *range(18200001, 18200007)]
            and seeds["evaluation_seeds"] == [*range(17100001, 17100025), *range(17200001, 17200025),
                                               *range(17300001, 17300025)]
            and seeds["fit_seeds"] == [40101, 40102, 40103]
            and seeds["hits"] == [] and isinstance(seeds["files"], dict) and seeds["files"]
            and len(seeds["files"]) == 232
            and seeds["scan_scope"] == "OTTO research documents and scripts, top-level frozen plans, all seed JSON/JSONL "
                                      "records and episodes.jsonl ledgers. Integer matches normalize underscores; current "
                                      "experiment proposals and sources excluded. No model, simulator or decoded numerical-array calls.",
            "scoped fresh TRAIN/EVAL/fit reservation")
    return native, prior, inherited, reference


def freeze(args):
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role))
        desc = descriptor(path)
        require(desc["sha256"] == getattr(args, f"{role}_sha256"), "external planning input")
        inputs[role] = {"path": str(path.relative_to(ROOT)), **desc}
    _, prior, _, _ = authenticate_inputs(inputs)
    names = set(prior["sources"]) | NEW_SOURCES
    plan = {"version": VERSION, "status": "frozen_before_collection", "configuration": CONFIGURATION,
            "limits": LIMITS, "call_caps": CALL_CAPS, "cohort": cohort(), "array_keys": sorted(ARRAY_KEYS),
            "payloads": sorted(PAYLOADS), "inputs": inputs, "native_inputs": prior["native_inputs"],
            "runtime": prior["runtime"], "sources": {name: descriptor(ROOT / name)["sha256"] for name in sorted(names)}}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "plan": descriptor(args.output)}))


def authenticate(args):
    require(args.plan.is_absolute() and descriptor(args.plan)["sha256"] == args.plan_sha256, "external collection plan")
    plan = read(args.plan)
    require(plan["version"] == VERSION and plan["status"] == "frozen_before_collection"
            and plan["configuration"] == CONFIGURATION and plan["limits"] == LIMITS and plan["call_caps"] == CALL_CAPS
            and plan["cohort"] == cohort() and plan["array_keys"] == sorted(ARRAY_KEYS)
            and set(plan["payloads"]) == PAYLOADS, "fixed collection allocation")
    for name, pin in plan["sources"].items():
        require(descriptor(ROOT / name)["sha256"] == pin, "collection source closure")
    native, prior, paths, reference = authenticate_inputs(plan["inputs"])
    require(set(plan["sources"]) == set(prior["sources"]) | NEW_SOURCES
            and all(plan["sources"][name] == pin for name, pin in prior["sources"].items())
            and plan["native_inputs"] == prior["native_inputs"] and plan["runtime"] == prior["runtime"],
            "unchanged native/source/runtime lineage")
    return plan, paths, reference, native


class Ledger:
    """Actual nested operations with durable attempts and return-before-ack."""

    def __init__(self, run):
        self.run, self.context, self.pending, self.sequence = run, {"phase": "setup"}, [], 0
        self.io_seconds, self.last_seconds = 0., {}
        self.calls = {name: {"attempted": 0, "returned": 0, "seconds": 0.} for name in CALL_CAPS}

    def emit(self, name, value):
        tick = time.perf_counter()
        try:
            with (self.run.out / name).open("a") as stream:
                stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            self.io_seconds += time.perf_counter() - tick

    def call(self, channel, function):
        self.run.check()
        require(channel in CALL_CAPS and self.calls[channel]["attempted"] < CALL_CAPS[channel], "collection call cap")
        self.sequence += 1
        event = {"call_id": self.sequence, "channel": channel, "context": dict(self.context),
                 "parent_call_id": None if not self.pending else self.pending[-1]["call_id"]}
        self.calls[channel]["attempted"] += 1
        self.pending.append(event)
        self.emit("work.jsonl", {"event": "attempt", **event})
        tick, io = time.perf_counter(), self.io_seconds
        result = function()
        raw, excluded = time.perf_counter() - tick, self.io_seconds - io
        require(raw >= excluded and self.pending[-1] is event, "nested operation timing")
        self.emit("work.jsonl", {"event": "return", **event, "seconds": raw - excluded,
                               "instrumented_seconds": raw, "excluded_io_seconds": excluded})
        self.calls[channel]["returned"] += 1
        self.calls[channel]["seconds"] += raw - excluded
        self.last_seconds[channel] = raw - excluded
        self.pending.pop()
        self.run.check()
        return result


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = self.active_actor = None
        self.ledger = Ledger(self)
        self.rows, self.offsets, self.total, self.queries = [], [0], 0, 0
        self.receipt = {"version": VERSION, "status": "started", "complete": False,
                        "training_updates": 0, "evaluation_episodes": 0, "efficacy_claim": False}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "shared collection deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "collection RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir()) <= LIMITS["output_bytes"], "collection output cap")

    def bind(self):
        require(descriptor(ROOT / CLOCK)["sha256"] == CLOCK_PIN, "clock pin before import")
        self.clock = load(ROOT / CLOCK, "_query_collection_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "actual parent launch missing")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                and self.launch["cap_seconds"] == LIMITS["native_seconds"]
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["native_seconds"] * 10**9
                and Path.cwd().resolve() == Path(self.launch["cwd"]).resolve() == ROOT, "original bounded collection parent")
        self.check()
        self.plan, self.paths, self.reference, self.native = authenticate(self.args)
        require(self.launch["watchdog_sha256"] == self.plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                and self.launch["clock_source_sha256"] == CLOCK_PIN, "original supervisor identity")
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=descriptor(self.args.supervision)["sha256"],
                            sources=self.plan["sources"], inputs=self.plan["inputs"], native_inputs=self.plan["native_inputs"],
                            limits=LIMITS)
        write(self.out / "started.json", {"request": {k: str(v) for k, v in vars(self.args).items()},
                                         "started_ns": self.start, "launch": self.launch, "clock_backend": self.clock.backend})

    def allocate(self):
        np = self.runtime.np
        return {"features": np.empty((MAX_ROWS, 31), np.float32), "labels": np.empty(MAX_ROWS, np.bool_),
                "neural_costs": np.empty((MAX_ROWS, 4), np.float32), "analytic_action": np.empty(MAX_ROWS, np.int64),
                "masks": np.empty((MAX_ROWS, 4), np.bool_), "scheduled_query": np.empty(MAX_ROWS, np.bool_),
                "neural_gap": np.empty(MAX_ROWS, np.float32)}

    def annotate(self, actor, policies, seen, queried):
        if queried:
            return seen["neural"].copy()
        before = (actor.public, actor.state.tobytes(), actor.last_choice, actor.progress, actor.belief.tobytes())
        _, scores = self.ledger.call("annotation_score", policies["neural"]._value_policy)
        after = (actor.public, actor.state.tobytes(), actor.last_choice, actor.progress, actor.belief.tobytes())
        require(before == after, "external annotation cannot change chosen action, filter or query history")
        return scores.copy()

    def episode(self, identity):
        runtime, np = self.runtime, self.runtime.np
        tick, io_start, start_row = time.perf_counter(), self.ledger.io_seconds, self.total
        initial_counts = {name: value["seconds"] for name, value in self.ledger.calls.items()}
        self.ledger.context = {"phase": "native_reset", **identity}
        config = {"Ndim": 2, "Ngrid": 53, "Nhits": 4, "lambda_over_dx": identity["sensing_length"],
                  "R_dt": 2., "norm_Poisson": "Euclidean"}
        env = self.ledger.call("native_reset", lambda: runtime.public.seeded_environment(
            runtime.source, identity["seed"], config, initial_hit=identity["initial_hit"]))
        require(env.p_Poisson.tobytes() == runtime.kernels[identity["regime"]].tobytes(), "exact native public kernel")
        current = self.reference.packet(runtime.public.observation(env, 0))
        recorder = self.reference.ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)
        policies, seen = {}, {}
        last_query, query_count, max_age = None, 0, 0
        age_counts = {}

        def score(name, view):
            require(view is self.active_actor.view, "bound readonly public score view")
            action, costs = policies[name]._value_policy()
            seen[name], seen[f"{name}_action"] = costs.copy(), int(action)
            return costs

        def gate(features, state):
            require(self.ledger.calls["tensorflow_value"]["attempted"] == before_tf,
                    "gate features precede current neural annotation")
            require(float(state[0]) == current["step"], "carried fixed schedule state")
            age = current["step"] if last_query is None else current["step"] - last_query
            require(float(features[self.indices["has_queried"]]) == float(last_query is not None)
                    and float(features[self.indices["query_age/2188"]]) == float(np.float32(age / HORIZON)),
                    "query age excludes every-state annotations")
            seen["features"], seen["age"] = features.copy(), age
            return scheduled_query(identity["schedule"], current["step"]), state + np.float32(1)

        self.ledger.context = {"phase": "actor_setup", **identity}
        actor = self.ledger.call("actor_construction", lambda: self.gate_class(
            current, runtime.kernels[identity["regime"]], identity["sensing_length"],
            lambda view: score("analytic", view), lambda view: score("neural", view), gate, state_size=1,
            check=self.check, emit=lambda event: self.ledger.emit("gate-operations.jsonl", {**identity, **event})))
        self.active_actor = actor

        def bind_backends():
            policies["analytic"] = self.analytic_policy(env=actor.view, model=None, sym_avg=False, allow_stay=False)
            policies["neural"] = runtime.policy(env=actor.view, model=recorder, sym_avg=True)

        self.ledger.call("backend_binding", bind_backends)
        witness = self.reference.belief_witness(actor, env, current, np)
        self.ledger.emit("public-transitions.jsonl", {"kind": "reset", **identity, "public": current,
                         "posterior": witness, "source_evaluation_only": env.source.tolist(), "draws": env.draw_log})
        tf_start = self.ledger.calls["tensorflow_value"]["returned"]
        for step in range(HORIZON):
            self.ledger.context = {"phase": "decision", **identity, "step": step}
            before_tf = self.ledger.calls["tensorflow_value"]["attempted"]
            action, _ = self.ledger.call("actor_choose", actor.choose)
            queried = actor.last_choice["queried"]
            require(queried == scheduled_query(identity["schedule"], step), "physical schedule choice")
            costs = self.annotate(actor, policies, seen, queried)
            require(self.ledger.calls["tensorflow_value"]["attempted"] - before_tf == 1,
                    "exactly one real neural annotation per TRAIN state")
            analytic_action = seen["analytic_action"]
            label, gap = annotation_label(costs, current["valid_actions"], analytic_action, np)
            expected_action = self.select_neural(costs, current["valid_actions"]) if queried else analytic_action
            require(action == expected_action, "annotation cannot change scheduled endpoint action")
            if queried:
                last_query, query_count = step, query_count + 1
            mask = np.asarray([a in current["valid_actions"] for a in range(4)], dtype=np.bool_)
            before_draws = len(env.draw_log)
            result = self.ledger.call("native_step", lambda action=action: env.step(action, quiet=True))
            public = self.reference.packet(runtime.public.observation(env, step + 1))
            require((int(result[0]), bool(result[2])) == (public["hit"], public["done"]), "actual native observation")
            self.ledger.call("actor_update", lambda action=action, public=public: actor.update(action, public))
            witness = self.reference.belief_witness(actor, env, public, np)
            require(actor.progress["pending_action"] is None and not actor.progress["pending_operations"], "actual final update")
            record = {"kind": "step", **identity, "row_index": self.total, "step": step + 1,
                      "public_before": current, "action": action, "public": public, "posterior": witness,
                      "scheduled_query": queried, "annotation_call": self.ledger.calls["tensorflow_value"]["returned"],
                      "annotation_route": "scheduled_query" if queried else "external_after_analytic_choice",
                      "features": seen["features"].tolist(), "query_age": seen["age"],
                      "state_after": actor.state.tolist(), "neural_costs": costs.tolist(),
                      "analytic_action": analytic_action, "label": label, "neural_gap": float(gap),
                      "draws": env.draw_log[before_draws:]}
            self.ledger.emit("public-transitions.jsonl", record)
            for key, value in (("features", seen["features"]), ("labels", label), ("neural_costs", costs),
                               ("analytic_action", analytic_action), ("masks", mask), ("scheduled_query", queried),
                               ("neural_gap", gap)):
                self.data[key][self.total] = value
            self.total += 1
            age_counts[seen["age"]] = age_counts.get(seen["age"], 0) + 1
            max_age = max(max_age, seen["age"])
            current = public
            if current["done"]:
                break
        count = self.total - start_row
        require(1 <= count <= HORIZON and (current["done"] or count == HORIZON)
                and self.ledger.calls["tensorflow_value"]["returned"] - tf_start == count
                and actor.progress["calls"]["neural_score"] == {"attempted": query_count, "returned": query_count},
                "complete uncropped path and exact annotations/queries")
        seconds = {name: value["seconds"] - initial_counts[name] for name, value in self.ledger.calls.items()}
        row = {**identity, "rows": count, "start_row": start_row, "end_row": self.total, "found": current["done"],
               "censored": not current["done"], "query_count": query_count, "annotation_calls": count,
               "external_annotation_calls": count - query_count, "maximum_query_age": max_age,
               "query_age_counts": age_counts, "operation_seconds": seconds, "wall_seconds": time.perf_counter() - tick,
               "journal_io_seconds": self.ledger.io_seconds - io_start, "gate_progress": actor.progress}
        self.ledger.emit("episodes.jsonl", row)
        self.rows.append(row)
        self.offsets.append(self.total)
        self.queries += query_count
        self.active_actor = None

    def save_data(self):
        arrays = {name: value[:self.total] for name, value in self.data.items()}
        arrays["episode_offsets"] = self.runtime.np.asarray(self.offsets, dtype=self.runtime.np.int64)
        require(set(arrays) == ARRAY_KEYS and len(self.offsets) == 61, "complete dataset schema")
        with (self.out / "training-data.npz").open("xb") as stream:
            self.runtime.np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return {name: {"shape": list(value.shape), "dtype": value.dtype.str,
                       "sha256_c_order": hashlib.sha256(value.tobytes(order="C")).hexdigest()} for name, value in arrays.items()}

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("collection terminated")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.bind()
            tick = time.perf_counter()
            self.runtime = self.reference.Run.setup(self)
            setup_seconds = time.perf_counter() - tick
            from openjev.research.otto_query_gate import FEATURE_NAMES, QueryGateActor
            from openjev.research.otto_reference_control import _SpaceAwarePolicy
            from openjev.research.otto_restricted_policy import select_inbounds_action

            self.gate_class, self.analytic_policy, self.select_neural = QueryGateActor, _SpaceAwarePolicy, select_inbounds_action
            self.indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
            self.data = self.ledger.call("dataset_allocate", self.allocate)
            tick = time.perf_counter()
            for identity in cohort():
                self.episode(identity)
            collection_seconds = time.perf_counter() - tick
            require(len(self.rows) == EPISODES and not self.ledger.pending
                    and all(v["attempted"] == v["returned"] for v in self.ledger.calls.values()), "all paths completed")
            for channel, count in (("native_reset", EPISODES), ("actor_construction", EPISODES), ("backend_binding", EPISODES),
                                   ("native_step", self.total), ("tensorflow_value", self.total), ("actor_choose", self.total),
                                   ("actor_update", self.total), ("annotation_score", self.total - self.queries)):
                require(self.ledger.calls[channel]["returned"] == count, f"exact collection channel {channel}")
            self.ledger.context = {"phase": "dataset_save"}
            arrays = self.ledger.call("dataset_save", self.save_data)
            costs = {"setup_wall_seconds": setup_seconds, "collection_wall_seconds": collection_seconds,
                     "dataset_save_seconds": self.ledger.last_seconds["dataset_save"],
                     "journal_io_seconds": self.ledger.io_seconds, "operation_seconds": self.ledger.calls,
                     "scope": "Raw physical collection costs include all annotations and gate/logger work. Nested calls overlap; "
                              "setup.json /192 allocation is inherited historical metadata, not this cohort allocation."}
            write(self.out / "collection-costs.json", costs)
            write(self.out / "summary.json", {"version": VERSION, "complete": True, "configuration": CONFIGURATION,
                  "episodes": self.rows, "rows": self.total, "queries": self.queries, "annotations": self.total,
                  "external_annotations": self.total - self.queries, "arrays": arrays, "feature_names": list(FEATURE_NAMES),
                  "costs": costs, "training_updates": 0, "evaluation_episodes": 0, "efficacy_claim": False})
            for name, pin in self.plan["sources"].items():
                require(descriptor(ROOT / name)["sha256"] == pin, "unchanged collection source")
            for record in self.plan["inputs"].values():
                require(descriptor(ROOT / record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged input")
            require(self.native.native_inputs(self.paths) == self.plan["native_inputs"], "unchanged loaded original inputs")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact completed collection inventory")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, calls=self.ledger.calls, pending=self.ledger.pending,
                                rows=self.total, completed_episodes=len(self.rows), queries=self.queries, annotations=self.total,
                                started_ns=self.start, finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                                requires_successful_original_supervisor=True,
                                scope="Complete TRAIN collection only, no fitting or EVAL; original successful parent also required.")
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc(),
                                calls=self.ledger.calls, pending=self.ledger.pending, completed_episodes=len(self.rows),
                                rows=self.total, active_gate=None if self.active_actor is None else self.active_actor.progress)
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                write(self.out / "failed.json", self.receipt)
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure if evidence publication also fails
                error.add_note(f"Failure publication: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    for role in sorted(ROLES):
        plan.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
        plan.add_argument(f"--{role.replace('_', '-')}-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / INTERPRETER, "qualified native interpreter only")
    require(args.output.is_absolute(), "absolute exclusive output required")
    if args.mode == "plan":
        freeze(args)
    else:
        Run(args).execute()


if __name__ == "__main__":
    main()
