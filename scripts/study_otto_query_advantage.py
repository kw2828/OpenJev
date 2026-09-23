"""Collect fresh public TRAIN anchors, then paired H32 query interventions.

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
VERSION = "otto-query-advantage-study-v1"
SELF, TEST = "scripts/study_otto_query_advantage.py", "tests/test_study_otto_query_advantage.py"
PROTOCOL = "research/otto-query-advantage-protocol.md"
COLLECTOR = "scripts/collect_otto_query_gate.py"
COLLECTOR_PIN = "5e2b06fc0ff5b888122e733f12b20c881deab7f56b94908ed5a283f7a48c76df"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
INTERPRETER = ".venv-otto-released-native/bin/python"
SCHEDULES, PREFIXES = ("always", "never", "period2"), (0, 4, 8, 16, 32)
HORIZON, EPISODES, MAX_ROWS, MAX_ANCHORS = 2188, 72, 157536, 360
LABEL_SEED, LABEL_HORIZON, REPLICAS = 19300001, 32, tuple(range(16))
FAILURE_RESERVE = 1024**2
CONFIGURATION = {"schedules": list(SCHEDULES), "regimes": {"base": 3., "shift": 4.},
    "first_seeds": {"base": 19100001, "shift": 19200001}, "cases_per_regime": 12,
    "initial_hit": "1 + case % 3", "rotation": "left by (regime_index * 12 + case) % 3",
    "horizon": HORIZON, "episodes": EPISODES, "maximum_rows": MAX_ROWS,
    "prefixes": list(PREFIXES), "anchor_id": "episode_index * 5 + prefix_index",
    "label_seed": LABEL_SEED, "label_horizon": LABEL_HORIZON, "replicate_ids": list(REPLICAS),
    "feature_dim": 31, "external_annotations": "only nonqueried available anchor states",
    "label_order": "all 72 paths and available anchors validated and saved before first source draw",
    "training_updates": 0, "evaluation_episodes": 0}
LIMITS = {"native_seconds": 900, "rss_bytes": 4 * 1024**3, "output_bytes": 6 * 1024**3,
          "non_sampler_output_bytes": 2 * 1024**3, "sampler_event_bytes": 1024, "sampler_events": 3019680,
          "native_resets": EPISODES, "native_steps": MAX_ROWS, "tensorflow_value_calls": 78888,
          "anchors": MAX_ANCHORS, "source_draws": 5760, "continuations": 11520, "paired_steps": 368640}
CALL_CAPS = {"tensorflow_construction": 1, "tensorflow_build": 1, "tensorflow_load": 1,
    "tensorflow_value": 78888, "native_reset": EPISODES, "native_step": MAX_ROWS,
    "actor_construction": EPISODES, "backend_binding": EPISODES, "actor_choose": MAX_ROWS,
    "actor_update": MAX_ROWS, "annotation_score": 120, "anchor_validate": MAX_ANCHORS,
    "anchor_save": 1, "pair_panel": MAX_ANCHORS, "panel_reduce": MAX_ANCHORS, "signal_reduce": 3}
SAMPLER_CAPS = {"anchor_snapshot": MAX_ANCHORS, "source_generator": 5760, "source_draw": 5760,
    "teacher_snapshot": 11520, "hit_generator": 11520, "teacher_choose": 357120,
    "movement": 368640, "hit_draw": 368640, "teacher_update": 368640}
ROLES = {"collection_plan", "sampler_plan", "sampler_receipt", "sampler_terminal", "sampler_review",
         "seed_review", "engineering_receipt"}
NEW_COMPONENTS = {SELF, TEST, "src/openjev/research/otto_query_pair_rollouts.py",
    "tests/test_otto_query_pair_rollouts.py", "src/openjev/research/otto_query_advantage.py",
    "tests/test_otto_query_advantage.py", "scripts/audit_otto_query_advantage.py",
    "tests/test_audit_otto_query_advantage.py"}
NEW_SOURCES = NEW_COMPONENTS | {PROTOCOL, COLLECTOR}
PAYLOADS = {"started.json", "runtime.json", "setup.json", "work.jsonl", "weights.jsonl", "forwards.jsonl",
    "gate-operations.jsonl", "public-transitions.jsonl", "native-truth.jsonl", "episodes.jsonl", "anchors.jsonl",
    "anchors.npz", "sampler-events.jsonl", "panels.jsonl", "costs.json", "summary.json"}
SAMPLER_PINS = {"sampler_plan": "7dd2f4a795a1772b68973d0fa2e507d97efe9ca4d7b18b2d663e7b889a909a4d",
    "sampler_receipt": "859bfb303daea27d187670b0e3e452855f2fd3be7ce5824c371a0526879268b6",
    "sampler_terminal": "785df84e5a9839903cd50f9b8b3e7d8a1a14ed21ec1c48e6230e41c723cb0c51",
    "sampler_review": "e1a1974f4e58b7844bc11d4e90549b9b14a09cadb62df9e1c6ef9b7febdd35cd"}


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
        for case in range(12):
            shift = (regime_index * 12 + case) % len(SCHEDULES)
            for schedule in SCHEDULES[shift:] + SCHEDULES[:shift]:
                seed = CONFIGURATION["first_seeds"][regime] + case
                result.append({"episode_id": f"train:{regime}:{seed}:{schedule}", "episode_index": len(result),
                               "regime": regime, "sensing_length": sensing, "case": case, "seed": seed,
                               "initial_hit": 1 + case % 3, "schedule": schedule})
    return result


def scheduled_query(schedule, step):
    require(schedule in SCHEDULES and type(step) is int and 0 <= step < HORIZON, "declared schedule/step")
    return schedule == "always" or schedule == "period2" and step % 2 == 0


def closed(directory, files, names):
    require(set(files) == set(names) and {p.name for p in directory.iterdir()} == set(names) | {"receipt.json"},
            "exact successful closed inventory")
    for name, desc in files.items():
        require(Path(name).name == name and descriptor(directory / name) == desc, "closed payload hash")


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact pilot input roles")
    paths = {}
    for role, record in inputs.items():
        require(set(record) == {"path", "sha256", "bytes"}, "input descriptor")
        paths[role] = regular(record["path"])
        require(descriptor(paths[role]) == {k: record[k] for k in ("sha256", "bytes")}, role)
        if role in SAMPLER_PINS:
            require(record["sha256"] == SAMPLER_PINS[role], "original sampler qualification")
    require(descriptor(ROOT / COLLECTOR)["sha256"] == COLLECTOR_PIN, "collector source before metadata import")
    collector = load(ROOT / COLLECTOR, "_advantage_inherited_collection")
    prior, inherited, reference, native = collector.authenticate(types.SimpleNamespace(
        plan=paths["collection_plan"], plan_sha256=inputs["collection_plan"]["sha256"]))
    sampler, receipt, terminal, review = (read(paths[k]) for k in
        ("sampler_plan", "sampler_receipt", "sampler_terminal", "sampler_review"))
    require(sampler["version"] == receipt["version"] == "otto-teacher-sampler-qualification-v1"
            and receipt["status"] == "completed" and receipt["qualified"] is True and receipt["pending"] is None
            and receipt["plan_sha256"] == inputs["sampler_plan"]["sha256"]
            and receipt["sources"] == sampler["sources"] and receipt["inputs"] == sampler["inputs"],
            "successful sampler source/input joins")
    closed(paths["sampler_receipt"].parent, receipt["files"], receipt["files"])
    for name, pin in sampler["sources"].items():
        require(descriptor(ROOT / name)["sha256"] == pin, "sampler source")
    for name, desc in sampler["inputs"].items():
        require(descriptor(ROOT / name) == desc, "sampler input")
    started = read(paths["sampler_receipt"].parent / "started.json")
    launch_path = regular(started["request"]["supervision"])
    launch = read(launch_path)
    require(descriptor(launch_path)["sha256"] == receipt["supervision_sha256"] and started["launch"] == launch
            and all(terminal[key] == value for key, value in launch.items())
            and terminal["status"] == "completed" and terminal["returncode"] == 0
            and terminal["timed_out"] is False and terminal["group_absent"] is True
            and terminal["cleanup"]["group_absent"] is True and terminal["cleanup"]["errors"] == []
            and terminal["error"] is terminal["clock_error"] is None and terminal["cap_seconds"] == 120
            and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"] <= terminal["finished_ns"]
            <= terminal["deadline_ns"], "original successful sampler parent")
    require(review["status"] == "passed" and review["failures"] == [] and review["checks"] == 38196
            and review["sources"] == sampler["sources"] and review["inputs"] == sampler["inputs"],
            "closed independent sampler qualification review")
    seeds = read(paths["seed_review"])
    require(seeds["version"] == "otto-query-advantage-seeds-v1" and seeds["status"] == "reserved_before_collection"
            and seeds["reserved_seeds"] == [*range(19100001, 19100013), *range(19200001, 19200013)]
            and seeds["label_seed"] == LABEL_SEED and seeds["bootstrap_seed"] == 19400001
            and seeds["hits"] == [] and isinstance(seeds["files"], dict) and seeds["files"], "fresh reserved TRAIN streams")
    engineering = read(paths["engineering_receipt"])
    require(engineering["status"] == "passed" and engineering["results"]
            and all(type(r["exit_code"]) is int and r["exit_code"] == 0 for r in engineering["results"])
            and engineering["sources_before"] == engineering["sources_after"]
            and set(engineering["sources_after"]) == NEW_COMPONENTS, "eight qualified new source/test files")
    for name, pin in engineering["sources_after"].items():
        require(descriptor(ROOT / name)["sha256"] == pin, "current qualified component bytes")
    closed(paths["engineering_receipt"].parent, engineering["files"], engineering["files"])
    sources = dict(prior["sources"])
    for name, pin in sampler["sources"].items():
        require(name not in sources or sources[name] == pin, "inherited source pin conflict")
        sources[name] = pin
    for name in NEW_SOURCES:
        pin = descriptor(ROOT / name)["sha256"]
        require(name not in sources or sources[name] == pin, "new source cannot overwrite inherited pin")
        sources[name] = pin
    return prior, sources, inherited, reference, native


def freeze(args):
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role)); desc = descriptor(path)
        require(desc["sha256"] == getattr(args, role + "_sha256"), "external planning pin")
        inputs[role] = {"path": str(path.relative_to(ROOT)), **desc}
    prior, sources, _, _, _ = authenticate_inputs(inputs)
    plan = {"version": VERSION, "status": "frozen_before_collection", "configuration": CONFIGURATION,
        "limits": LIMITS, "call_caps": CALL_CAPS, "sampler_caps": SAMPLER_CAPS, "cohort": cohort(),
        "payloads": sorted(PAYLOADS), "inputs": inputs, "native_inputs": prior["native_inputs"],
        "runtime": prior["runtime"], "sources": sources}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "plan": descriptor(args.output)}))


def authenticate(args):
    require(args.plan.is_absolute() and descriptor(args.plan)["sha256"] == args.plan_sha256, "external pilot plan")
    plan = read(args.plan)
    require(plan["version"] == VERSION and plan["status"] == "frozen_before_collection"
            and plan["configuration"] == CONFIGURATION and plan["limits"] == LIMITS
            and plan["call_caps"] == CALL_CAPS and plan["sampler_caps"] == SAMPLER_CAPS
            and plan["cohort"] == cohort() and set(plan["payloads"]) == PAYLOADS, "fixed prospective pilot")
    prior, sources, paths, reference, native = authenticate_inputs(plan["inputs"])
    require(plan["sources"] == sources and plan["native_inputs"] == prior["native_inputs"]
            and plan["runtime"] == prior["runtime"], "exact complete source/runtime lineage")
    return plan, paths, reference, native


class Ledger:
    """Actual nested operations with durable attempts and return-before-ack."""

    def __init__(self, run):
        self.run, self.context, self.pending, self.sequence = run, {"phase": "setup"}, [], 0
        self.io_seconds, self.last_seconds = 0., {}
        self.pending_publications, self.publication_sequence = [], 0
        self.calls = {name: {"attempted": 0, "returned": 0, "seconds": 0.} for name in CALL_CAPS}

    def emit(self, name, value):
        tick = time.perf_counter()
        self.publication_sequence += 1
        intent = {"publication_id": self.publication_sequence, "file": name,
                  "context": json.loads(json.dumps(self.context)),
                  "event": value.get("event"), "kind": value.get("kind"),
                  "anchor_id": value.get("anchor_id"), "episode_id": value.get("episode_id")}
        self.pending_publications.append(intent)
        try:
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            intent["bytes"] = len(payload.encode())
            self.run.guard_write(name, intent["bytes"])
            with (self.run.out / name).open("a") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self.pending_publications.pop()
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
        self.rows, self.total, self.queries, self.external = [], 0, 0, 0
        self.anchors, self.anchor_arrays, self.panels = [], [], []
        self.sampler_calls = {name: {"attempted": 0, "returned": 0} for name in SAMPLER_CAPS}
        self.sampler_pending, self.sampler_records, self.sampler_events = [], 0, 0
        self.reserved_output_bytes = 0
        self.receipt = {"version": VERSION, "status": "started", "complete": False,
                        "training_updates": 0, "evaluation_episodes": 0, "efficacy_claim": False}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "shared collection deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "collection RSS cap")
        sizes = {p.name: p.stat().st_size for p in self.out.iterdir()}
        require(sum(sizes.values()) <= LIMITS["output_bytes"]
                and sum(v for k, v in sizes.items() if k != "sampler-events.jsonl") <= LIMITS["non_sampler_output_bytes"],
                "total and non-sampler output caps")

    def guard_write(self, name, size):
        self.check()
        sizes = {p.name: p.stat().st_size for p in self.out.iterdir()}
        reserve = self.reserved_output_bytes + FAILURE_RESERVE
        require(sum(sizes.values()) + size + reserve <= LIMITS["output_bytes"], "projected total output")
        if name != "sampler-events.jsonl":
            require(sum(v for k, v in sizes.items() if k != "sampler-events.jsonl") + size + reserve
                    <= LIMITS["non_sampler_output_bytes"], "projected non-sampler output")

    def publish(self, name, value):
        self.ledger.publication_sequence += 1
        intent = {"publication_id": self.ledger.publication_sequence, "file": name,
                  "context": json.loads(json.dumps(self.ledger.context)), "kind": "json_artifact"}
        self.ledger.pending_publications.append(intent)
        size = len((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        intent["bytes"] = size
        self.guard_write(name, size)
        write(self.out / name, value)
        self.ledger.pending_publications.pop()

    def failure_publish(self, name, value, spent):
        # Failure evidence uses pre-reserved bytes even after the deadline.
        size = len((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        sizes = {p.name: p.stat().st_size for p in self.out.iterdir()}
        require(spent + size <= FAILURE_RESERVE and sum(sizes.values()) + size <= LIMITS["output_bytes"]
                and sum(v for k, v in sizes.items() if k != "sampler-events.jsonl") + size
                    <= LIMITS["non_sampler_output_bytes"], "bounded reserved failure evidence")
        write(self.out / name, value)
        return spent + size

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
        self.publish("started.json", {"request": {k: str(v) for k, v in vars(self.args).items()},
                                         "started_ns": self.start, "launch": self.launch, "clock_backend": self.clock.backend})

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
        last_query, query_count, max_age, external_count = None, 0, 0, 0
        available_steps = []
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
                         "posterior": witness})
        self.ledger.emit("native-truth.jsonl", {"kind": "reset", **identity,
                         "source_evaluation_only": env.source.tolist(), "draws": env.draw_log})
        tf_start = self.ledger.calls["tensorflow_value"]["returned"]
        for step in range(HORIZON):
            self.ledger.context = {"phase": "decision", **identity, "step": step}
            before_tf = self.ledger.calls["tensorflow_value"]["attempted"]
            action, _ = self.ledger.call("actor_choose", actor.choose)
            queried = actor.last_choice["queried"]
            require(queried == scheduled_query(identity["schedule"], step), "physical schedule choice")
            analytic_action = seen["analytic_action"]
            is_anchor = step in PREFIXES
            costs = None
            if is_anchor:
                costs = self.annotate(actor, policies, seen, queried)
                external_count += int(not queried)
                self.capture_anchor(identity, step, actor, seen, costs)
                available_steps.append(step)
            require(self.ledger.calls["tensorflow_value"]["attempted"] - before_tf == int(queried or is_anchor),
                    "queries plus only skipped-anchor annotations")
            expected_action = self.select_neural(seen["neural"], current["valid_actions"]) if queried else analytic_action
            require(action == expected_action, "annotation cannot change scheduled endpoint action")
            if queried:
                last_query, query_count = step, query_count + 1
            before_draws = len(env.draw_log)
            result = self.ledger.call("native_step", lambda action=action: env.step(action, quiet=True))
            public = self.reference.packet(runtime.public.observation(env, step + 1))
            require((int(result[0]), bool(result[2])) == (public["hit"], public["done"]), "actual native observation")
            self.ledger.call("actor_update", lambda action=action, public=public: actor.update(action, public))
            witness = self.reference.belief_witness(actor, env, public, np)
            require(actor.progress["pending_action"] is None and not actor.progress["pending_operations"], "actual final update")
            record = {"kind": "step", **identity, "row_index": self.total, "step": step + 1,
                      "public_before": current, "action": action, "public": public, "posterior": witness,
                      "scheduled_query": queried, "anchor": is_anchor,
                      "external_annotation": is_anchor and not queried,
                      "features": seen["features"].tolist(), "query_age": seen["age"],
                      "state_after": actor.state.tolist(), "analytic_action": analytic_action}
            self.ledger.emit("public-transitions.jsonl", record)
            self.ledger.emit("native-truth.jsonl", {"kind": "step", **identity, "step": step + 1,
                             "draws": env.draw_log[before_draws:]})
            self.total += 1
            age_counts[seen["age"]] = age_counts.get(seen["age"], 0) + 1
            max_age = max(max_age, seen["age"])
            current = public
            if current["done"]:
                break
        count = self.total - start_row
        require(1 <= count <= HORIZON and (current["done"] or count == HORIZON)
                and self.ledger.calls["tensorflow_value"]["returned"] - tf_start == query_count + external_count
                and actor.progress["calls"]["neural_score"] == {"attempted": query_count, "returned": query_count},
                "complete uncropped path and exact annotations/queries")
        seconds = {name: value["seconds"] - initial_counts[name] for name, value in self.ledger.calls.items()}
        row = {**identity, "rows": count, "start_row": start_row, "end_row": self.total, "found": current["done"],
               "censored": not current["done"], "query_count": query_count,
               "neural_calls": query_count + external_count, "external_annotation_calls": external_count,
               "available_anchor_steps": available_steps, "maximum_query_age": max_age,
               "query_age_counts": age_counts, "operation_seconds": seconds, "wall_seconds": time.perf_counter() - tick,
               "journal_io_seconds": self.ledger.io_seconds - io_start, "gate_progress": actor.progress}
        for prefix_index, prefix in enumerate(PREFIXES):
            if prefix not in available_steps:
                require(current["done"] and prefix >= count, "missing anchor only after found")
                missing = {**identity, "anchor_id": identity["episode_index"] * 5 + prefix_index,
                           "prefix_index": prefix_index, "preaction_step": prefix,
                           "status": "unavailable_found", "found_after_moves": count}
                self.ledger.emit("anchors.jsonl", missing)
                self.anchors.append(missing)
        self.ledger.emit("episodes.jsonl", row)
        self.rows.append(row)
        self.queries += query_count
        self.external += external_count
        self.active_actor = None

    def capture_anchor(self, identity, step, actor, seen, costs):
        np = self.runtime.np
        public, belief = actor.public, actor.belief
        prefix_index = PREFIXES.index(step)
        record = {**identity, "anchor_id": identity["episode_index"] * 5 + prefix_index,
                  "prefix_index": prefix_index, "preaction_step": step, "status": "available",
                  "array_index": len(self.anchor_arrays), "public": public,
                  "belief_sha256": hashlib.sha256(belief.tobytes(order="C")).hexdigest(),
                  "belief_mass": float(np.sum(belief, dtype=np.float64)),
                  "features": seen["features"].tolist(), "analytic_action": seen["analytic_action"],
                  "neural_action": self.select_neural(costs, public["valid_actions"]),
                  "analytic_costs": [float(x) if np.isfinite(x) else None for x in seen["analytic"]],
                  "neural_costs": costs.tolist()}
        def validate():
            try:
                return self.snapshot(public, belief, self.runtime.kernels[identity["regime"]])
            except (ValueError, TypeError, FloatingPointError) as error:
                # Only snapshot-domain errors mean unsupported, not a later
                # caller budget or journal failure after successful validation.
                record.update(status="unsupported", error=repr(error))
                try:
                    self.ledger.emit("anchors.jsonl", record)
                except BaseException as secondary:  # noqa: BLE001 - the snapshot-domain failure remains primary
                    error.add_note(f"Unsupported-anchor publication also failed: {secondary!r}")
                raise
        self.ledger.call("anchor_validate", validate)
        owned = np.frombuffer(belief.tobytes(order="C"), dtype=np.float64).reshape(53, 53)
        features = np.frombuffer(seen["features"].tobytes(), dtype=np.float32)
        self.ledger.emit("anchors.jsonl", record)
        self.anchor_arrays.append((owned, features))
        self.anchors.append(record)

    def save_anchors(self):
        np = self.runtime.np
        available = [r for r in self.anchors if r["status"] == "available"]
        require(len(self.rows) == EPISODES and len(self.anchors) == MAX_ANCHORS
                and len({r["anchor_id"] for r in self.anchors}) == MAX_ANCHORS
                and len(available) == len(self.anchor_arrays), "complete anchor-slot accounting before labels")
        arrays = {"beliefs": np.stack([a[0] for a in self.anchor_arrays]),
                  "features": np.stack([a[1] for a in self.anchor_arrays]),
                  "anchor_ids": np.asarray([r["anchor_id"] for r in available], dtype=np.int64)}
        require(len(available) <= MAX_ANCHORS and all(a.flags.c_contiguous for a in arrays.values()), "bounded anchor archive")
        self.guard_write("anchors.npz", sum(a.nbytes for a in arrays.values()) + 4096)
        with (self.out / "anchors.npz").open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return descriptor(self.out / "anchors.npz")

    def sampler_event(self, anchor_id, event):
        require(event["anchor_id"] == anchor_id and event["seed"] == LABEL_SEED, "sampler outer identity")
        require(self.sampler_events < 3019680 and
                len((json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()) <= 1024,
                "fixed sampler serialization bound")
        kind = event["event"]
        if kind in ("attempt", "return"):
            name, key = event["operation"], (anchor_id, event["operation_id"])
            require(name in SAMPLER_CAPS, "declared sampler operation")
            count = self.sampler_calls[name]
            if kind == "attempt":
                require(count["attempted"] < SAMPLER_CAPS[name], "sampler operation cap")
                count["attempted"] += 1
                self.sampler_pending.append({"anchor_id": anchor_id, "operation_id": key[1], "operation": name})
            else:
                require(self.sampler_pending and self.sampler_pending[-1] == {
                    "anchor_id": anchor_id, "operation_id": key[1], "operation": name}, "sampler return identity")
            self.ledger.emit("sampler-events.jsonl", event)
            if kind == "return":
                count["returned"] += 1
                self.sampler_pending.pop()
        else:
            self.ledger.emit("sampler-events.jsonl", event)
            if kind == "record":
                self.sampler_records += 1
        self.sampler_events += 1

    def label_anchors(self):
        require(len(self.rows) == EPISODES and len(self.anchors) == MAX_ANCHORS
                and (self.out / "anchors.npz").is_file() and not self.ledger.pending,
                "all physical paths and durable validated anchors precede source sampling")
        for anchor in self.anchors:
            if anchor["status"] != "available":
                continue
            self.ledger.context = {"phase": "paired_labels", "anchor_id": anchor["anchor_id"],
                                   "episode_id": anchor["episode_id"]}
            belief, _ = self.anchor_arrays[anchor["array_index"]]
            records = self.ledger.call("pair_panel", lambda anchor=anchor, belief=belief: self.sample_pair(
                anchor["public"], belief, self.runtime.kernels[anchor["regime"]],
                analytic_action=anchor["analytic_action"], neural_action=anchor["neural_action"],
                seed=LABEL_SEED, anchor_id=anchor["anchor_id"], replicate_ids=REPLICAS,
                horizon=LABEL_HORIZON, check=self.check,
                emit=lambda event, anchor=anchor: self.sampler_event(anchor["anchor_id"], event)))
            reduced = self.ledger.call("panel_reduce", lambda anchor=anchor, records=records:
                self.reduce_panel(records, analytic_action=anchor["analytic_action"],
                    neural_action=anchor["neural_action"], replicate_ids=REPLICAS, horizon=LABEL_HORIZON))
            panel = {"episode_id": anchor["episode_id"], "anchor_id": anchor["anchor_id"], "reduction": reduced}
            self.ledger.emit("panels.jsonl", panel)
            self.panels.append(panel)
        require(not self.sampler_pending and all(v["attempted"] == v["returned"] for v in self.sampler_calls.values()),
                "every sampler attempt returned")
        require(len(self.panels) == len(self.anchor_arrays)
                and self.sampler_records == sum(p["reduction"]["physical_record_count"] for p in self.panels),
                "complete distinct-action label panels")
        moves = sum(r["steps"] for p in self.panels for r in p["reduction"]["records"])
        found = sum(r["found"] for p in self.panels for r in p["reduction"]["records"])
        for name, count in {"anchor_snapshot": len(self.panels), "source_generator": 16 * len(self.panels),
                "source_draw": 16 * len(self.panels), "teacher_snapshot": self.sampler_records,
                "hit_generator": self.sampler_records, "teacher_choose": moves - self.sampler_records,
                "movement": moves, "hit_draw": moves - found, "teacher_update": moves}.items():
            require(self.sampler_calls[name] == {"attempted": count, "returned": count}, "exact sampler count: " + name)
        summaries = {}
        for group in ("all", "base", "shift"):
            episodes = [e["episode_id"] for e in self.rows if group == "all" or e["regime"] == group]
            rows = [p for p in self.panels if p["episode_id"] in episodes]
            summaries[group] = self.ledger.call("signal_reduce", lambda rows=rows, episodes=episodes:
                                               self.reduce_signal(rows, episode_ids=episodes))
        return summaries

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("pilot terminated")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.bind()
            for name in PAYLOADS:
                if name.endswith(".jsonl"):
                    with (self.out / name).open("x"):
                        pass
            tick = time.perf_counter()
            # The unchanged setup writes two small JSON objects directly. Hold
            # 1 MiB for both while its nested ledger writes use the same budget.
            require(len(json.dumps(self.plan["runtime"]).encode()) < 256 * 1024, "bounded setup metadata")
            self.guard_write("setup.json", 1024**2)
            self.reserved_output_bytes = 1024**2
            self.runtime = self.reference.Run.setup(self)
            require(sum((self.out / n).stat().st_size for n in ("runtime.json", "setup.json"))
                    <= self.reserved_output_bytes, "inherited setup fits its reserved JSON space")
            self.reserved_output_bytes = 0
            from openjev.research.otto_query_advantage import summarize_advantage, summarize_signal
            from openjev.research.otto_query_gate import FEATURE_NAMES, QueryGateActor
            from openjev.research.otto_query_pair_rollouts import sample_query_pair
            from openjev.research.otto_reference_control import _SpaceAwarePolicy
            from openjev.research.otto_restricted_policy import select_inbounds_action
            from openjev.research.otto_teacher_snapshot import TeacherSnapshot

            self.gate_class, self.analytic_policy, self.select_neural = QueryGateActor, _SpaceAwarePolicy, select_inbounds_action
            self.snapshot, self.sample_pair = TeacherSnapshot, sample_query_pair
            self.reduce_panel, self.reduce_signal = summarize_advantage, summarize_signal
            self.indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
            setup_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            for identity in cohort():
                self.episode(identity)
            collection_seconds = time.perf_counter() - tick
            require(len(self.rows) == EPISODES and not self.ledger.pending
                    and all(v["attempted"] == v["returned"] for v in self.ledger.calls.values()), "all complete paths")
            for channel, count in (("native_reset", EPISODES), ("actor_construction", EPISODES),
                ("backend_binding", EPISODES), ("native_step", self.total), ("actor_choose", self.total),
                ("actor_update", self.total), ("tensorflow_value", self.queries + self.external),
                ("annotation_score", self.external), ("anchor_validate", len(self.anchor_arrays))):
                require(self.ledger.calls[channel]["returned"] == count, "exact collection count: " + channel)
            self.ledger.context = {"phase": "save_all_anchors_before_labels"}
            anchor_file = self.ledger.call("anchor_save", self.save_anchors)
            tick = time.perf_counter()
            summaries = self.label_anchors()
            label_seconds = time.perf_counter() - tick
            costs = {"setup_wall_seconds": setup_seconds, "collection_wall_seconds": collection_seconds,
                     "paired_labels_wall_seconds": label_seconds,
                     "anchor_save_seconds": self.ledger.last_seconds["anchor_save"],
                     "journal_io_seconds": self.ledger.io_seconds, "operation_seconds": self.ledger.calls,
                     "sampler_counts": self.sampler_calls,
                     "scope": "Physical phases include checks and logging. Nested operations overlap; do not add them. "
                              "Historical setup.json /192 allocation is not used for this pilot."}
            self.publish("costs.json", costs)
            self.publish("summary.json", {"version": VERSION, "complete": True,
                  "configuration": CONFIGURATION, "episodes": self.rows, "trajectory_rows": self.total,
                  "queries": self.queries, "external_annotations": self.external,
                  "anchor_slots": len(self.anchors), "available_anchors": len(self.anchor_arrays),
                  "unavailable_anchors": len(self.anchors) - len(self.anchor_arrays), "anchors_file": anchor_file,
                  "signals": summaries, "costs": costs, "training_updates": 0, "evaluation_episodes": 0,
                  "scope": "Technical completion only. Independent saved audit owns the frozen signal admission rule. "
                           "One forced endpoint action then common analytic continuation, not repeated neural-policy value."})
            require(authenticate(self.args)[0] == self.plan, "final unchanged source/input/runtime closure")
            require(self.native.native_inputs(self.paths) == self.plan["native_inputs"], "unchanged original loaded inputs")
            require(descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged launch")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact completed pilot inventory")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, calls=self.ledger.calls,
                pending=self.ledger.pending, pending_publications=list(self.ledger.pending_publications),
                sampler_calls=self.sampler_calls, sampler_pending=self.sampler_pending,
                sampler_records=self.sampler_records, sampler_events=self.sampler_events,
                completed_episodes=len(self.rows), trajectory_rows=self.total,
                available_anchors=len(self.anchor_arrays), panels=len(self.panels), queries=self.queries,
                external_annotations=self.external, started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9, requires_successful_original_supervisor=True)
            self.publish("receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc(),
                calls=self.ledger.calls, pending=self.ledger.pending, pending_publications=self.ledger.pending_publications,
                sampler_calls=self.sampler_calls,
                sampler_pending=self.sampler_pending, sampler_records=self.sampler_records, sampler_events=self.sampler_events,
                completed_episodes=len(self.rows), trajectory_rows=self.total, panels=len(self.panels),
                available_anchors=len(self.anchor_arrays), active_gate=None if self.active_actor is None else self.active_actor.progress)
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                spent = self.failure_publish("failed.json", self.receipt, 0)
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                self.failure_publish("receipt.json", self.receipt, spent)
            except BaseException as secondary:  # noqa: BLE001 - keep the first failure when evidence publication also fails
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
