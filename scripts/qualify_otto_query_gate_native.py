"""Bounded original-TensorFlow query-gate integration, never efficacy or training.

Plan/authentication is stdlib-only. Numerical setup and ForwardRecorder are
reused unchanged from the qualified reference runner after complete lineage
authentication. Actual run requires its original bounded supervisor.
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
VERSION = "otto-query-gate-native-qualification-v1"
SELF = "scripts/qualify_otto_query_gate_native.py"
BOUNDARY = "scripts/study_otto_boundary_control.py"
REFERENCE = "scripts/study_otto_released_reference.py"
CLOCK = "src/openjev/research/suspend_clock.py"
GATE = "src/openjev/research/otto_query_gate.py"
GATE_TEST = "tests/test_otto_query_gate.py"
PROTOCOL = "research/otto-query-gate-native-protocol.md"
INTERPRETER = ".venv-otto-released-native/bin/python"
PINS = {
    BOUNDARY: "f00af83597e9ecd4bf8e2354fad7e4c6e25f69a9e182fcdd12f0e47050081dd8",
    REFERENCE: "da2a7e418dd29819bddf190f46d1707f1163a22b01a048b33d3dedab875ad87e",
    CLOCK: "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
    GATE: "3ef019d408934dc5277378998713e6b38c35334eafcd9b17db163e3e5f8b124c",
    GATE_TEST: "fe77fd7fb5c1a853716236b93fb8734575fbada4d034224e03f05bb5bad1d889",
}
MODES = ("always_query", "never_query", "alternating")
CONFIGURATION = {
    "modes": list(MODES), "regimes": {"base": 3., "shift": 4.},
    "first_seeds": {"base": 1110001, "shift": 1120001}, "initial_hits": [1, 2, 3],
    "steps_per_case": 16, "episodes": 18, "Ngrid": 53, "Nhits": 4, "Ndim": 2,
    "R_dt": 2., "norm_Poisson": "Euclidean", "symmetry_average": True,
    "alternating": "query at preaction even counter, beginning zero; carry counter+1",
    "score_belief_action_parity": "exact bytes and exact qualified selection",
    "actual_neural_forwards_per_queried_step": 2, "never_query_neural_forwards": 0,
    "training_updates": 0, "efficacy_claim": False,
}
LIMITS = {"native_seconds": 180, "rss_bytes": 4 * 1024**3, "output_bytes": 64 * 1024**2,
          "native_resets": 18, "native_steps": 288, "tensorflow_value_calls": 288}
CALL_CAPS = {"tensorflow_construction": 1, "tensorflow_build": 1, "tensorflow_load": 1,
             "tensorflow_value": 288, "native_reset": 18, "native_step": 288,
             "actor_construction": 54, "backend_binding": 18, "actor_choose": 288,
             "actor_update": 864, "analytic_reference_score": 288, "neural_reference_score": 144}
ROLES = {"boundary_plan", "gate_engineering", "seed_review"}
NEW_SOURCES = {SELF, GATE, GATE_TEST, PROTOCOL}
PAYLOADS = {"started.json", "runtime.json", "setup.json", "qualification-costs.json", "work.jsonl",
            "weights.jsonl", "forwards.jsonl", "gate-operations.jsonl", "features.jsonl", "checks.jsonl",
            "episodes.jsonl", "summary.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    value = Path(value)
    path = value if value.is_absolute() else ROOT / value
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
            and path.is_relative_to(ROOT) and ".." not in path.parts, "contained regular file")
    return path


def descriptor(path):
    path = regular(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


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


def source_names(prior):
    return set(prior["sources"]) | set(PINS) | NEW_SOURCES


def native_inputs(paths):
    return {name: {"path": str(path.relative_to(ROOT)), **descriptor(path)} for name, path in sorted(paths.items())}


def authenticate_inputs(inputs):
    require(set(inputs) == ROLES, "exact metadata input roles")
    paths = {}
    for role, record in inputs.items():
        require(set(record) == {"path", "sha256", "bytes"}, "input descriptor schema")
        paths[role] = regular(record["path"])
        require(descriptor(paths[role]) == {k: record[k] for k in ("sha256", "bytes")}, f"input: {role}")
    for name, pin in PINS.items():
        require(descriptor(ROOT / name)["sha256"] == pin, f"immutable helper: {name}")
    # This chain checks original native TF qualification, analytic and restricted
    # preflights, previous successful process closure and complete runtime.
    boundary = load(ROOT / BOUNDARY, "_query_native_boundary_auth")
    prior, inherited = boundary.authenticate(types.SimpleNamespace(
        plan=paths["boundary_plan"], plan_sha256=inputs["boundary_plan"]["sha256"]))
    engineering = read(paths["gate_engineering"])
    require(engineering["status"] == "passed" and engineering["source_before"] == engineering["source_after"]
            == {name: PINS[name] for name in (GATE, GATE_TEST)}
            and len(engineering["commands"]) == 2
            and all(type(row["returncode"]) is int and row["returncode"] == 0 for row in engineering["commands"]),
            "passed current-byte fabricated engineering")
    directory = paths["gate_engineering"].parent
    require({p.name for p in directory.iterdir()} == set(engineering["files"]) | {paths["gate_engineering"].name},
            "closed engineering attempt")
    for name, desc in engineering["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == desc, "engineering payload")
    seeds = read(paths["seed_review"])
    require(seeds["version"] == "otto-query-gate-seed-reservation-v1"
            and seeds["status"] == "reserved_before_native_run"
            and seeds["reserved_seeds"] == [1110001, 1110002, 1110003, 1120001, 1120002, 1120003]
            and seeds["hits"] == [] and isinstance(seeds["files"], dict) and len(seeds["files"]) == 220
            and seeds["scan_scope"] == "OTTO research protocols, scripts, top-level frozen plans, and seed JSON records; "
                                      "integer tokens normalized for underscores; excludes this new experiment. "
                                      "Namespace collision check, not every historical step log.",
            "cleared exact six case scoped seed reservation")
    return boundary.PRIOR, prior, inherited


def authenticate(args):
    require(args.plan.is_absolute() and descriptor(args.plan)["sha256"] == args.plan_sha256, "external frozen plan")
    plan = read(args.plan)
    require(plan["version"] == VERSION and plan["status"] == "frozen_before_native_run"
            and plan["configuration"] == CONFIGURATION and plan["limits"] == LIMITS
            and plan["call_caps"] == CALL_CAPS and set(plan["payloads"]) == PAYLOADS, "fixed mechanical allocation")
    for name, pin in plan["sources"].items():
        require(descriptor(ROOT / name)["sha256"] == pin, f"source: {name}")
    reference, prior, paths = authenticate_inputs(plan["inputs"])
    require(set(plan["sources"]) == source_names(prior)
            and all(plan["sources"][name] == pin for name, pin in prior["sources"].items())
            and plan["runtime"] == prior["runtime"] and plan["native_inputs"] == native_inputs(paths),
            "complete unchanged lineage, loaded inputs and runtime")
    return plan, paths, reference


def freeze(args):
    inputs = {}
    for role in sorted(ROLES):
        path = regular(getattr(args, role))
        desc = descriptor(path)
        require(desc["sha256"] == getattr(args, f"{role}_sha256"), f"external planning input: {role}")
        inputs[role] = {"path": str(path.relative_to(ROOT)), **desc}
    _, prior, paths = authenticate_inputs(inputs)
    plan = {"version": VERSION, "status": "frozen_before_native_run", "configuration": CONFIGURATION,
            "limits": LIMITS, "call_caps": CALL_CAPS, "payloads": sorted(PAYLOADS),
            "inputs": inputs, "native_inputs": native_inputs(paths), "runtime": prior["runtime"],
            "sources": {name: descriptor(ROOT / name)["sha256"] for name in sorted(source_names(prior))}}
    write(args.output, plan)
    print(json.dumps({"status": plan["status"], "plan": descriptor(args.output)}))


class Ledger:
    """ForwardRecorder-compatible bounded nested journal, return-before-ack."""

    def __init__(self, run):
        self.run, self.context, self.pending, self.sequence = run, {"phase": "setup"}, [], 0
        self.io_seconds = 0.
        self.last_seconds = {}
        self.calls = {channel: {"attempted": 0, "returned": 0, "seconds": 0.} for channel in CALL_CAPS}

    def emit(self, name, value):
        tick = time.perf_counter()
        try:
            with (self.run.out / name).open("a") as stream:
                stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            self.io_seconds += time.perf_counter() - tick

    def call(self, channel, function):
        self.run.check()
        require(channel in CALL_CAPS and self.calls[channel]["attempted"] < CALL_CAPS[channel], "declared call cap")
        self.sequence += 1
        event = {"call_id": self.sequence, "channel": channel, "context": dict(self.context),
                 "parent_call_id": None if not self.pending else self.pending[-1]["call_id"]}
        self.calls[channel]["attempted"] += 1
        self.pending.append(event)
        self.emit("work.jsonl", {"event": "attempt", **event})
        tick, io = time.perf_counter(), self.io_seconds
        result = function()
        raw, excluded = time.perf_counter() - tick, self.io_seconds - io
        elapsed = raw - excluded
        require(elapsed >= 0 and self.pending[-1] is event, "nested actual operation timing")
        self.emit("work.jsonl", {"event": "return", **event, "seconds": elapsed,
                               "instrumented_seconds": raw, "excluded_io_seconds": excluded})
        self.calls[channel]["returned"] += 1
        self.calls[channel]["seconds"] += elapsed
        self.last_seconds[channel] = elapsed
        self.pending.pop()
        self.run.check()
        return result


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.ledger, self.active_actor = Ledger(self), None
        self.receipt = {"version": VERSION, "status": "started", "qualified": False,
                        "training_updates": 0, "external_model_calls": 0, "efficacy_claim": False}
        self.rows, self.queries, self.steps, self.comparisons = [], 0, 0, 0

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "shared native deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "native RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir()) <= LIMITS["output_bytes"], "native output cap")

    def same(self, left, right, message):
        self.comparisons += 1
        require(left == right, message)

    def bind(self):
        require(descriptor(ROOT / CLOCK)["sha256"] == PINS[CLOCK], "clock pin before import")
        self.clock = load(ROOT / CLOCK, "_query_native_clock").SuspendClock()
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
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 180 * 10**9
                and Path.cwd().resolve() == Path(self.launch["cwd"]).resolve() == ROOT, "original bounded parent")
        self.check()
        self.plan, self.paths, self.reference = authenticate(self.args)
        require(self.launch["watchdog_sha256"] == self.plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                and self.launch["clock_source_sha256"] == PINS[CLOCK], "actual supervisor source identity")
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=descriptor(self.args.supervision)["sha256"],
                            sources=self.plan["sources"], inputs=self.plan["inputs"], native_inputs=self.plan["native_inputs"],
                            limits=LIMITS)
        write(self.out / "started.json", {"request": {k: str(v) for k, v in vars(self.args).items()},
              "started_ns": self.start, "launch": self.launch, "clock_backend": self.clock.backend})

    def state_check(self, actors, env, public):
        witnesses = [self.reference.belief_witness(actor, env, public, self.runtime.np) for actor in actors]
        self.same(witnesses[0], witnesses[1], "gate/restricted posterior witness")
        self.same(witnesses[0], witnesses[2], "gate/analytic posterior witness")
        return witnesses[0]

    def episode(self, regime, seed, hit, mode):
        runtime, np = self.runtime, self.runtime.np
        identity = {"regime": regime, "seed": seed, "initial_hit": hit, "mode": mode}
        self.ledger.context = {"phase": "native_reset", **identity}
        config = {"Ndim": 2, "Ngrid": 53, "Nhits": 4, "lambda_over_dx": CONFIGURATION["regimes"][regime],
                  "R_dt": 2., "norm_Poisson": "Euclidean"}
        env = self.ledger.call("native_reset", lambda: runtime.public.seeded_environment(
            runtime.source, seed, config, initial_hit=hit))
        self.same(env.p_Poisson.tobytes(), runtime.kernels[regime].tobytes(), "exact supplied native kernel")
        current = self.reference.packet(runtime.public.observation(env, 0))
        recorder = self.reference.ForwardRecorder(runtime.model, runtime.policy._value_policy.__code__, self.ledger, np)
        policies, seen = {}, {}

        def score(name, view):
            require(view is self.active_actor.view, "bound score-only readonly view")
            predicted, costs = policies[name]._value_policy()
            seen[name] = costs.copy()
            seen[f"{name}_action"] = int(predicted)
            return costs

        def gate(features, state):
            self.same(self.ledger.calls["tensorflow_value"]["attempted"], before_tf, "gate runs before current neural score")
            self.same(float(state[0]), float(current["step"]), "carried alternating counter")
            self.same(float(features[self.feature_indices["has_queried"]]), float(last_query is not None), "query history flag")
            age = current["step"] if last_query is None else current["step"] - last_query
            self.same(float(features[self.feature_indices["query_age/2188"]]), float(np.float32(age / 2188)), "query age")
            queried = mode == "always_query" or mode == "alternating" and int(state[0]) % 2 == 0
            next_state = state + np.float32(1)
            self.ledger.emit("features.jsonl", {**identity, "step": current["step"], "features": features.tolist(),
                 "state_before": state.tolist(), "state_after": next_state.tolist(), "queried": queried,
                 "tensorflow_calls_at_gate": before_tf})
            return queried, next_state

        self.ledger.context = {"phase": "actor_setup", **identity}
        actor = self.ledger.call("actor_construction", lambda: self.gate_class(
            current, runtime.kernels[regime], CONFIGURATION["regimes"][regime],
            lambda view: score("analytic", view), lambda view: score("neural", view), gate,
            state_size=1, check=self.check,
            emit=lambda event: self.ledger.emit("gate-operations.jsonl", {**identity, **event})))
        self.active_actor = actor

        def bind_backends():
            policies["analytic"] = self.analytic_policy(env=actor.view, model=None, sym_avg=False, allow_stay=False)
            policies["neural"] = runtime.policy(env=actor.view, model=recorder, sym_avg=True)

        self.ledger.call("backend_binding", bind_backends)
        neural = self.ledger.call("actor_construction", lambda: self.restricted(
            current, runtime.kernels[regime], recorder, runtime.policy, sym_avg=True))
        analytic = self.ledger.call("actor_construction", lambda: runtime.analytic(
            current, runtime.kernels[regime], allow_stay=False))
        actors = (actor, neural, analytic)
        witness = self.state_check(actors, env, current)
        self.ledger.emit("checks.jsonl", {"kind": "reset", **identity, "public": current,
                         "posterior": witness, "source_evaluation_only": env.source.tolist(), "draws": env.draw_log})
        last_query, queries = None, 0
        tf_start = self.ledger.calls["tensorflow_value"]["returned"]
        for step in range(1, 17):
            self.ledger.context = {"phase": "decision", **identity, "step": step}
            before_tf = self.ledger.calls["tensorflow_value"]["attempted"]
            action, costs = self.ledger.call("actor_choose", actor.choose)
            queried = actor.last_choice["queried"]
            self.same(queried, mode == "always_query" or mode == "alternating" and step % 2 == 1, "fixed gate schedule")
            ref_action, ref_costs = self.ledger.call("analytic_reference_score", analytic._policy._value_policy)
            self.same(seen["analytic"].tobytes(), ref_costs.tobytes(), "independent exact analytic raw costs")
            self.same(seen["analytic_action"], int(ref_action), "independent analytic backend action on every decision")
            neural_costs = None
            if queried:
                _, neural_costs = self.ledger.call("neural_reference_score", neural._policy._value_policy)
                expected = self.select_neural(neural_costs, current["valid_actions"])
                self.same(costs.tobytes(), neural_costs.tobytes(), "independent exact neural raw costs")
                self.same(action, expected, "original restricted neural selected action")
                last_query, queries = current["step"], queries + 1
            else:
                self.same(action, int(ref_action), "independent analytic selected action")
                self.same(costs.tobytes(), ref_costs.tobytes(), "unmodified analytic output")
            self.same(self.ledger.calls["tensorflow_value"]["attempted"] - before_tf, 2 * int(queried),
                      "two actual original forwards iff queried, none otherwise")
            self.same(neural._pending_action, None, "reference neural score did not commit counterfactual action")
            self.same(analytic._pending_action, None, "reference analytic score did not commit counterfactual action")
            before_draws = len(env.draw_log)
            result = self.ledger.call("native_step", lambda action=action: env.step(action, quiet=True))
            public = self.reference.packet(runtime.public.observation(env, step))
            self.same((int(result[0]), bool(result[2])), (public["hit"], public["done"]), "native public event")
            for actual in actors:
                self.ledger.call("actor_update", lambda actual=actual, public=public, action=action: actual.update(action, public))
            witness = self.state_check(actors, env, public)
            self.same(actor.progress["pending_action"], None, "actual gate action assimilated")
            self.same(actor.progress["calls"]["neural_score"], {"attempted": queries, "returned": queries}, "gate query accounting")
            self.ledger.emit("checks.jsonl", {"kind": "step", **identity, "step": step, "action": action,
                "queried": queried, "cost_dtype": str(costs.dtype),
                "selected_costs": [float(x) if np.isfinite(x) else None for x in costs],
                "analytic_costs": [float(x) if np.isfinite(x) else None for x in ref_costs],
                "neural_costs": None if neural_costs is None else neural_costs.tolist(),
                "public": public, "posterior": witness, "draws": env.draw_log[before_draws:], "passed": True})
            self.steps += 1
            current = public
            if current["done"]:
                break
        forwards = self.ledger.calls["tensorflow_value"]["returned"] - tf_start
        self.same(forwards, queries * 2, "complete mode-specific TF count")
        require(not actor.progress["failed"] and not actor.progress["pending_operations"], "complete gate lifecycle")
        row = {**identity, "steps": current["step"], "found": current["done"], "queries": queries,
               "tensorflow_value_calls": forwards, "gate_progress": actor.progress,
               "source_evaluation_only": env.source.tolist(), "scope": "short integration path, not efficacy"}
        self.ledger.emit("episodes.jsonl", row)
        self.queries += queries
        self.rows.append(row)
        self.active_actor = None

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("qualification terminated")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.bind()
            # Unmodified original setup, including exact eight-tensor identity.
            tick = time.perf_counter()
            self.runtime = self.reference.Run.setup(self)
            setup_wall = time.perf_counter() - tick
            from openjev.research.otto_query_gate import FEATURE_NAMES, QueryGateActor
            from openjev.research.otto_reference_control import _SpaceAwarePolicy
            from openjev.research.otto_restricted_policy import RestrictedPolicyActor, select_inbounds_action

            self.gate_class, self.analytic_policy = QueryGateActor, _SpaceAwarePolicy
            self.restricted, self.select_neural = RestrictedPolicyActor, select_inbounds_action
            self.feature_indices = {name: i for i, name in enumerate(FEATURE_NAMES)}
            tick = time.perf_counter()
            for regime in CONFIGURATION["regimes"]:
                for hit in (1, 2, 3):
                    seed = CONFIGURATION["first_seeds"][regime] + hit - 1
                    for mode in MODES:
                        self.episode(regime, seed, hit, mode)
            mechanical_wall = time.perf_counter() - tick
            require(len(self.rows) == 18 and not self.ledger.pending
                    and all(v["attempted"] == v["returned"] for v in self.ledger.calls.values()), "complete prescribed mechanical work")
            self.same(self.ledger.calls["native_reset"]["returned"], 18, "all18 reset calls")
            self.same(self.ledger.calls["native_step"]["returned"], self.steps, "actual total step count")
            self.same(self.ledger.calls["tensorflow_value"]["returned"], 2 * self.queries, "actual original TF total")
            costs = {"setup_wall_seconds": setup_wall, "mechanical_wall_seconds": mechanical_wall,
                     "scope": "Actual physical intervals. setup.json retains historical /192 allocation metadata only; "
                              "no new per-episode allocation, speed or efficacy estimate. Nested call times overlap."}
            write(self.out / "qualification-costs.json", costs)
            write(self.out / "summary.json", {"version": VERSION, "qualified": True, "configuration": CONFIGURATION,
                  "episodes": self.rows, "native_steps": self.steps, "queries": self.queries,
                  "calls": self.ledger.calls, "comparisons": self.comparisons, "costs": costs,
                  "feature_names": list(FEATURE_NAMES), "efficacy_claim": False})
            for name, pin in self.plan["sources"].items():
                require(descriptor(ROOT / name)["sha256"] == pin, "unchanged frozen source")
            for record in self.plan["inputs"].values():
                require(descriptor(ROOT / record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged input")
            require(native_inputs(self.paths) == self.plan["native_inputs"], "unchanged actual loaded native inputs")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged plan/launch")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact completed output inventory")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", qualified=True, files=files, calls=self.ledger.calls,
                pending=self.ledger.pending, native_steps=self.steps, queries=self.queries, comparisons=self.comparisons,
                completed_episodes=18, started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished - self.start) / 1e9, requires_successful_original_supervisor=True,
                scope="Transition/readout/query integration only. Overall acceptance also requires original parent exit0, "
                      "no timeout and absent process group; no learning or efficacy admission.")
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", qualified=False, error=repr(error), traceback=traceback.format_exc(),
                calls=self.ledger.calls, pending=self.ledger.pending, completed_episodes=len(self.rows),
                native_steps=self.steps, queries=self.queries, comparisons=self.comparisons,
                active_gate=None if self.active_actor is None else self.active_actor.progress)
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                write(self.out / "failed.json", self.receipt)
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve primary failure despite publication failure
                error.add_note(f"Failure publication: {secondary!r}")
            raise


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
    require(Path(sys.executable).absolute() == ROOT / INTERPRETER, "qualified native interpreter only")
    require(args.output.is_absolute(), "absolute exclusive output required")
    if args.mode == "plan":
        freeze(args)
    else:
        Run(args).execute()


if __name__ == "__main__":
    main()
