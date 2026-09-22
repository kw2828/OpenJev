"""Strict native generation qualification, never teacher-policy evaluation.

Plan mode reads source/package/engineering metadata only. Run mode requires an
externally pinned plan and the original suspend-inclusive supervisor. It keeps
every numerical mismatch and completes the fixed checklist unless an operation
or resource check raises. Overall acceptance also needs the later parent terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-teacher-sampler-qualification-v1"
SCRIPT = "scripts/qualify_otto_teacher_sampler.py"
PROTOCOL = "research/otto-teacher-sampler-qualification-protocol.md"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
NATIVE = "tmp/otto-source-review-01/isotropic/classes/sourcetracking.py"
REFERENCE_PLAN = "output/otto-spatial-control-v1/plan-01.json"
REFERENCE_PIN = "fdbcd35a5a8ac990f1221ceb5e6a661f62e04b5d5cf264df6aca2b9224880c51"
ENVIRONMENT = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
LIMITS = {"native_seconds": 120, "rss_bytes": 4 * 1024**3, "output_bytes": 64 * 1024**2}
PINS = {
    NATIVE: "1057f129fa8a3249a7297a9ef2c15059d37430b12e42a2f1d43dfcdffe3e76f1",
    CLOCK: "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
    SUPERVISOR: "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144",
    "src/openjev/research/otto_public.py": "438631a18005493e0cafa0777315e2158cac97cc7d0e71ad9fd6402284615b3d",
    "src/openjev/research/otto_released_policy.py": "8abac34fef2a6517d43209d5ca27514633363248f66a249e2f4d6615aa0a0d45",
    "src/openjev/research/otto_reference_control.py": "21e5f6ece09426d26c4a3214d95f0f4803a8efd4bacb9ca347c1d896407c5893",
    "src/openjev/research/otto_teacher_snapshot.py": "c318bd1d743b1589153c360c4bd1a2afdec5df34e7b769be0b2c3a26a87e9c58",
    "src/openjev/research/otto_teacher_rollouts.py": "0fb4f0e4e45e0bed0c40c90c3bff5b19f58b888a838b1f155ab4ff6f0a777612",
    "src/openjev/research/otto_teacher_costs.py": "39f00522aaa2fd8b88231a74f23f4db72d2e8fc86dcb68b83de26af2f35e30fe",
    "tests/test_otto_teacher_rollouts.py": "10cd8d4576bc6b62f24ce4b2a2c2cfc7852d89f65cbef7af0e26849887dbf470",
    "tests/test_otto_teacher_snapshot.py": "d4a26616b6f43b8a56cffeb373c0a4d4280c1d02a92d025e394b3f3777f56f7c",
    "tests/test_otto_teacher_costs.py": "9c9539f45fcfd4c5359b24b399b35945f4d1a42294af6630746de8231030ed4e",
}
SOURCES = (*PINS, SCRIPT, PROTOCOL, "tests/test_qualify_otto_teacher_sampler.py",
           "src/openjev/__init__.py", "src/openjev/research/__init__.py")
PREREQUISITES = {
    "output/otto-teacher-rollouts-engineering-v1/attempt-01/receipt.json":
        "ea9c185ef82370fc0d682f618b7f0ea20a89315a84fd86c2a8dac84d0ce9ed2d",
    "output/otto-teacher-snapshot-engineering-v1/attempt-02/receipt.json":
        "75b9b55b9a72a617161414219f5825838462f5c62a3fee3eeec24ba0e86adfc4",
    "output/otto-teacher-cost-engineering-v1/attempt-01/receipt.json":
        "044aa77818c5648f5f54eb2afd06d58a1bb16f5a1c59c7becea14c5d7b25f9a4",
}
GEOMETRIES = (
    ((26, 26), 0, (24, 26)), ((26, 26), 1, (29, 27)),
    ((26, 26), 2, (52, 52)), ((26, 26), 3, (0, 0)),
    ((0, 0), 0, (52, 52)), ((0, 0), 2, (1, 2)),
    ((52, 52), 1, (0, 0)), ((52, 52), 3, (51, 52)),
    ((0, 0), 1, (0, 52)), ((0, 0), 3, (2, 1)),
    ((52, 52), 0, (52, 0)), ((52, 52), 2, (50, 51)),
)
DIRECT_VECTORS = ((0., .25, 0., .75), (.25, 0., .75, 0.), (0., 0., 0., 1.), (1., 0., 0., 0.))
EXPECTED = {"native_construction": 3, "state_installation": 408, "native_step": 408,
            "snapshot_construction": 408, "snapshot_update": 408, "direct_hit_draw": 44,
            "direct_source_draw": 9, "helper_hit_vector": 396, "helper_cdf": 449,
            "helper_category": 440, "helper_source_draw": 9, "helper_movement": 276,
            "native_hit_draw": 440, "native_source_draw": 12}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
            "absolute paths without symbolic links required")


def digest(path):
    regular(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configuration():
    return {"regimes": [3, 4, 5], "mechanical_seeds": [0, 1, 2], "Ngrid": 53, "Nhits": 4,
            "R_dt": 2., "initial_hit": 1, "anchor_step": 7, "geometries": json.loads(json.dumps(GEOMETRIES)),
            "direct_vectors": [list(x) for x in DIRECT_VECTORS], "probes_per_geometry": 11,
            "native_steps": 408, "direct_categories": 44, "direct_sources": 9,
            "acceptance": "Exact finite bytes/values and indices; all planned comparisons required"}


def metadata():
    """No numerical import, empirical array, model or native operation."""
    sources = {name: digest(ROOT / name)["sha256"] for name in SOURCES}
    require(all(sources[name] == pin for name, pin in PINS.items()), "unchanged reviewed sources")
    require(digest(ROOT / REFERENCE_PLAN)["sha256"] == REFERENCE_PIN, "original runtime reference")
    old = read(ROOT / REFERENCE_PLAN)
    runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
               "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    require(all(runtime[k] == old[k] for k in runtime), "same existing spatial-control runtime")
    inputs = {REFERENCE_PLAN: digest(ROOT / REFERENCE_PLAN)}
    for name, pin in PREREQUISITES.items():
        path = ROOT / name
        require(digest(path)["sha256"] == pin, "original component qualification receipt")
        receipt = read(path)
        require(receipt["status"] == "passed" and receipt["results"]
                and all(type(x["exit_code"]) is int and x["exit_code"] == 0 for x in receipt["results"]),
                "passing component qualification")
        require(receipt["sources_before"] == receipt["sources_after"]
                and all(sources[p] == sha for p, sha in receipt["sources_after"].items()),
                "component results bind current source bytes")
        require({p.name for p in path.parent.iterdir()} == set(receipt["files"]) | {"receipt.json"},
                "closed component evidence")
        inputs[name] = digest(path)
        for basename, record in receipt["files"].items():
            require(Path(basename).name == basename, "flat prerequisite filename")
            child = path.parent / basename
            require(digest(child) == record, "prerequisite payload identity")
            inputs[str(child.relative_to(ROOT))] = record
    return {"version": VERSION, "status": "frozen", "configuration": configuration(),
            "limits": LIMITS, "expected_calls": EXPECTED, "environment": ENVIRONMENT,
            "sources": sources, "inputs": inputs, **runtime}


class Tape:
    """One explicit scalar uniform; an unexpected or found-path draw raises."""

    def __init__(self, uniform):
        self.uniform, self.calls = uniform, 0

    def random(self):
        if self.uniform is None or self.calls:
            raise RuntimeError("uniform tape exhausted or forbidden found-path draw")
        self.calls += 1
        return self.uniform


def oracle_cdf(values, np):
    running = np.float64(0)
    cumulative = []
    for value in values:
        running = np.float64(running + np.float64(value))
        cumulative.append(running)
    require(np.isfinite(running) and running > 0, "positive finite reference CDF mass")
    return np.asarray([np.float64(x / running) for x in cumulative]), float(running)


def probes(cumulative, np):
    upper = float(np.nextafter(np.float64(1), np.float64(0)))
    values = [0., upper]
    for boundary in cumulative[:3]:
        values.extend(min(upper, float(x)) for x in
                      (np.nextafter(boundary, np.float64(0)), boundary,
                       np.nextafter(boundary, np.float64(1))))
    return values


def select_oracle(cumulative, uniform):
    return next(i for i, boundary in enumerate(cumulative) if float(boundary) > uniform)


def public_packet(position, hit=1, done=False, step=7):
    position = tuple(int(x) for x in position)
    allowed = () if done else tuple(a for a in range(4)
        if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53)
    return {"position": position, "hit": int(hit), "done": bool(done), "step": step,
            "valid_actions": allowed}


def anchor(position, np):
    x, y = np.indices((53, 53), dtype=np.int64)
    belief = (1 + ((17 * x + 29 * y + 7 * x * y) % 97)).astype(np.float64)
    belief[position] = 0
    return belief / np.sum(belief, dtype=np.float64)


def install(env, position, source, belief, np):
    env.agent, env.source, env.p_source = list(position), np.asarray(source, dtype=int), belief.copy()
    env.obs = {"hit": 1, "done": False}
    env.hit_map = np.full((53, 53), -1, dtype=int)
    env.hit_map[position] = 1
    env.cumulative_hits, env.agent_near_boundaries, env.agent_stuck = 0, 0, False
    env._agento, env._agentoo, env._repeated_visits = [0, 0], [53, 53], 0
    env.entropy = env._entropy(env.p_source)


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.calls, self.operation_stack, self.rows = {}, [], []
        self.receipt = {"version": VERSION, "status": "started", "qualified": False,
                        "limits": LIMITS, "teacher_choices": 0, "learned_model_calls": 0,
                        "training_calls": 0, "sample_teacher_panel_calls": 0,
                        "requires_successful_original_supervisor": True}

    @property
    def pending(self):
        return self.operation_stack[-1] if self.operation_stack else None

    def check(self):
        now = self.clock.now_ns()
        deadline = self.launch["deadline_ns"] if self.launch else self.start + 120 * 10**9
        require(now < deadline, "suspend-inclusive qualification deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "qualification RSS bound")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS["output_bytes"],
                "qualification output bound")

    def call(self, name, context, function, *args, record=None, **kwargs):
        self.check()
        require(name in EXPECTED, "declared operation")
        if self.pending is not None:
            require(name in ("native_hit_draw", "native_source_draw")
                    and self.pending["operation"] in ("native_construction", "native_step",
                                                       "direct_hit_draw", "direct_source_draw"),
                    "only actual native draws may nest in their enclosing operation")
        count = self.calls.setdefault(name, {"attempted": 0, "returned": 0})
        require(count["attempted"] < EXPECTED[name], "prospective operation cap")
        count["attempted"] += 1
        frame = {"operation": name, "ordinal": count["attempted"], "context": context,
                 "parent": None if self.pending is None else {
                     "operation": self.pending["operation"], "ordinal": self.pending["ordinal"]}}
        self.operation_stack.append(frame)
        append(self.out / "work.jsonl", {"event": "attempt", **frame})
        result = function(*args, **kwargs)
        evidence = {} if record is None else record(result)
        append(self.out / "work.jsonl", {"event": "return", **frame, **evidence})
        count["returned"] += 1
        require(self.operation_stack[-1] is frame, "nested operation returned to its caller")
        self.operation_stack.pop()
        self.check()
        return result

    def recorded_native(self, original):
        """Log the original bound draw, including construction's source draw."""
        owner = self

        class RecordedSourceTracking(original):
            def __init__(self, *args, **kwargs):
                # seeded_environment has already installed local RNGs/logs.
                # Capture its bound _draw before delegating native initialization.
                original_draw = self._draw

                def draw(channel, probabilities):
                    require(channel in ("source", "hit"), "no initial-hit draw in fixed qualification")

                    def returned(index):
                        saved = self.draw_log[-1]
                        require(saved["channel"] == channel and saved["selected_index"] == index,
                                "returned original draw evidence")
                        return {"draw": saved, "native_draw_counts": dict(self._public_draw_counts)}

                    return owner.call(f"native_{channel}_draw", dict(owner.pending["context"]),
                                      original_draw, channel, probabilities, record=returned)

                self._draw = draw
                super().__init__(*args, **kwargs)

        return RecordedSourceTracking

    def compare(self, row):
        row["passed"] = all(row["checks"].values())
        append(self.out / "comparisons.jsonl", row)
        self.rows.append({"id": row["id"], "passed": row["passed"]})

    def arrays(self, name, **values):
        self.check()
        with (self.out / name).open("xb") as stream:
            self.np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        self.check()

    def bind(self):
        require(digest(ROOT / CLOCK)["sha256"] == PINS[CLOCK], "reviewed clock bytes")
        self.clock = load(ROOT / CLOCK, "_teacher_sampler_clock").SuspendClock()
        self.start = self.clock.now_ns()
        require(self.clock.backend in ("mach_continuous_time", "CLOCK_BOOTTIME"), "native clock required")
        require(digest(self.args.plan)["sha256"] == self.args.plan_sha256, "external plan pin")
        self.plan = read(self.args.plan)
        require(self.plan == metadata(), "complete current plan/source/runtime identity")
        require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "pre-import one-thread settings")
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch receipt missing")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        expected = [sys.executable, str(ROOT / SCRIPT), "run", "--plan", str(self.args.plan),
                    "--plan-sha256", self.args.plan_sha256, "--supervision", str(self.args.supervision),
                    "--output", str(self.out)]
        require(command == expected and command == [sys.executable, *sys.argv], "exact original invocation")
        require(self.launch["pid"] == os.getpid() and self.launch["pgid"] == os.getpgrp()
                and self.launch["parent_pid"] == os.getppid(), "actual original process identity")
        require(self.launch["version"] == "dialogue-observation-supervision-v2"
                and self.launch["cap_seconds"] == 120 and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 120 * 10**9
                and self.launch["cwd"] == str(ROOT) == str(Path.cwd())
                and self.launch["watchdog_sha256"] == PINS[SUPERVISOR]
                and self.launch["clock_source_sha256"] == PINS[CLOCK], "bounded original supervisor")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=self.plan["sources"],
                            inputs=self.plan["inputs"], supervision_sha256=digest(self.args.supervision)["sha256"])
        write(self.out / "started.json", {"version": VERSION, "request": {k: str(v) for k, v in vars(self.args).items()},
              "started_ns": self.start, "clock_backend": self.clock.backend, "launch": self.launch})
        self.check()

    def vector_check(self, context, probability, native_record, uniform):
        np, sampler = self.np, self.sampler
        original = np.asarray(native_record["probabilities"], dtype=np.float64)
        reference_cdf, reference_mass = oracle_cdf(original, np)
        copied, actual_cdf, actual_mass = self.call("helper_cdf", context, sampler._categorical, probability)
        index = self.call("helper_category", context, sampler.categorical_index, probability, uniform)
        oracle = select_oracle(reference_cdf, uniform)
        checks = {"probability": copied.tobytes() == original.tobytes(),
                  "cdf": actual_cdf.tobytes() == reference_cdf.tobytes(),
                  "cdf_mass": actual_mass == reference_mass == native_record["cdf_mass"],
                  "uniform": uniform == native_record["uniform"],
                  "index": index == oracle == native_record["selected_index"],
                  "support": bool(original[oracle] > 0 and copied[index] > 0)}
        return {"checks": checks, "native_draw": native_record, "helper_probability": copied.tolist(),
                "native_cdf": reference_cdf.tolist(), "helper_cdf": actual_cdf.tolist(),
                "helper_cdf_mass": actual_mass, "helper_index": index, "oracle_index": oracle}

    def transition(self, env, regime, geometry, position, action, source, uniform, probe, found=False):
        np, sampler = self.np, self.sampler
        context = {"regime": regime, "geometry": geometry, "probe": probe, "found_fixture": found}
        number = self.calls.get("native_step", {}).get("attempted", 0)
        belief = self.anchors[position]
        self.call("state_installation", context, install, env, position, source, belief, np)
        initial = public_packet(position)
        actor = self.call("snapshot_construction", context, sampler.TeacherSnapshot, initial, belief, env.p_Poisson)
        tape = Tape(uniform)
        env._public_rngs["hit"] = tape
        before = len(env.draw_log)
        returned = self.call("native_step", context, env.step, action, hit=None, quiet=True)
        native_public = self.public.observation(env, 8)._asdict()
        native_public["position"] = tuple(native_public["position"])
        moved = list(position)
        moved[action // 2] = max(0, min(52, moved[action // 2] + 2 * (action % 2) - 1))
        checks = {"native_move": tuple(env.agent) == tuple(moved),
                  "found": bool(returned[2]) == found, "end_mass": float(returned[1]) == float(found)}
        if action in initial["valid_actions"]:
            movement = self.call("helper_movement", context, sampler._movement, position, action, source)
            checks["helper_movement"] = movement == {"position": moved, "found": found}
        else:
            existing_move, possible = sampler._move(action, position)
            checks["blocked_transport_only"] = not possible and existing_move == moved
        if found:
            row = {"checks": checks}
            checks.update(no_draw=len(env.draw_log) == before and tape.calls == 0, sentinel=int(returned[0]) == -2)
            helper_public = public_packet(moved, -2, True, 8)
        else:
            require(len(env.draw_log) == before + 1 and tape.calls == 1, "exactly one native hit draw")
            native_draw = env.draw_log[-1]
            probability = self.call("helper_hit_vector", context, sampler.source_hit_probabilities,
                                    actor, source, moved)
            row = self.vector_check(context, probability, native_draw, uniform)
            row["checks"].update(checks)
            checks = row["checks"]
            checks["native_hit"] = int(returned[0]) == native_draw["selected_index"]
            helper_public = public_packet(moved, row["helper_index"], False, 8)
        checks["public_packet"] = helper_public == native_public
        # Native returned observation isolates public-filter parity even if the
        # separately retained sampled-category comparison failed.
        self.call("snapshot_update", context, actor.update, action, native_public)
        public_belief = actor.belief
        checks["posterior"] = public_belief.tobytes() == env.p_source.tobytes()
        checks["updated_packet"] = actor.public == native_public
        if found:
            point_mass = np.zeros((53, 53), dtype=np.float64)
            point_mass[tuple(moved)] = 1.
            checks["terminal_point_mass"] = (env.p_source.tobytes() == point_mass.tobytes()
                                               == public_belief.tobytes())
        self.arrays(f"posterior-{number:03d}.npz", native=env.p_source, public=public_belief)
        row.update(id=f"step-{number:03d}", **context, position=list(position), action=action,
                   source_evaluation_only=list(source), native_public=native_public, helper_public=helper_public,
                   native_return=[int(returned[0]), float(returned[1]), bool(returned[2])],
                   posterior_file=f"posterior-{number:03d}.npz", native_draw_count=len(env.draw_log)-before)
        self.compare(row)
        return row.get("native_draw")

    def body(self):
        require(not any(n in sys.modules for n in ("numpy", "scipy", "torch", "tensorflow", "tf_keras")),
                "clean numerical import boundary")
        import numpy as np

        self.np = np
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research import otto_public, otto_teacher_rollouts

        self.public, self.sampler = otto_public, otto_teacher_rollouts
        native = self.recorded_native(load(ROOT / NATIVE, "_teacher_sampler_original_native").SourceTracking)
        self.anchors = {q: anchor(q, np) for q in ((26, 26), (0, 0), (52, 52))}
        self.arrays("anchors.npz", center=self.anchors[(26, 26)], lower=self.anchors[(0, 0)],
                    upper=self.anchors[(52, 52)])
        environments = []
        for seed, regime in enumerate((3, 4, 5)):
            config = {"Ndim": 2, "Ngrid": 53, "Nhits": 4, "lambda_over_dx": float(regime),
                      "R_dt": 2., "norm_Poisson": "Euclidean", "draw_source": True}
            env = self.call("native_construction", {"regime": regime}, otto_public.seeded_environment,
                            native, seed, config, initial_hit=1)
            environments.append(env)
            require(env.N == 53 and env.Nhits == 4 and env.Nactions == 4, "fixed native geometry")
            require(env._public_draw_counts == {"initial": 0, "source": 1, "hit": 0}, "one initialization source")
            append(self.out / "initializations.jsonl", {"regime": regime, "mechanical_seed": seed,
                   "public": otto_public.observation(env, 0)._asdict(), "unused_source_draw": env.draw_log[0]})
            self.arrays(f"kernel-lambda{regime}.npz", likelihood=env.p_Poisson)
            for geometry, (position, action, source) in enumerate(GEOMETRIES):
                first = self.transition(env, regime, geometry, position, action, source, 0., 0)
                cumulative, _ = oracle_cdf(first["probabilities"], np)
                for probe, uniform in enumerate(probes(cumulative, np)[1:], 1):
                    self.transition(env, regime, geometry, position, action, source, uniform, probe)
            for action in range(4):
                source = [26, 26]
                source[action // 2] += 2 * (action % 2) - 1
                self.transition(env, regime, f"found-{action}", (26, 26), action, tuple(source), None, 0, True)
        env = environments[0]
        for vector_id, vector in enumerate(DIRECT_VECTORS):
            cumulative, _ = oracle_cdf(vector, np)
            for probe, uniform in enumerate(probes(cumulative, np)):
                context = {"vector": vector_id, "probe": probe}
                tape = Tape(uniform)
                env._public_rngs["hit"] = tape
                index = self.call("direct_hit_draw", context, env._draw, "hit", vector)
                row = self.vector_check(context, vector, env.draw_log[-1], uniform)
                row["checks"]["one_direct_draw"] = tape.calls == 1 and index == env.draw_log[-1]["selected_index"]
                row.update(id=f"category-{vector_id}-{probe}", **context)
                self.compare(row)
        source_beliefs = [np.zeros((53, 53), dtype=np.float64) for _ in range(2)] + [self.anchors[(26, 26)]]
        source_beliefs[0][0, 0] = 1.
        source_beliefs[1][0, 52], source_beliefs[1][52, 0] = .25, .75
        for fixture, belief in enumerate(source_beliefs):
            for probe, uniform in enumerate((0., .5, float(np.nextafter(np.float64(1), np.float64(0))))):
                context = {"source_fixture": fixture, "probe": probe}
                env.p_source, env.agent = belief.copy(), [26, 26]
                tape = Tape(uniform)
                env._public_rngs["source"] = tape
                self.call("direct_source_draw", context, env._draw_a_source)
                actual = self.call("helper_source_draw", context, self.sampler._source_draw, Tape(uniform), belief)
                original = env.draw_log[-1]
                vector = np.asarray(original["probabilities"], dtype=np.float64)
                reference_cdf, mass = oracle_cdf(vector, np)
                copied, helper_cdf, helper_mass = self.call("helper_cdf", context, self.sampler._categorical,
                                                          belief.reshape(-1, order="C"))
                index = select_oracle(reference_cdf, uniform)
                source = [index // 53, index % 53]
                checks = {"probability": copied.tobytes() == vector.tobytes(),
                          "cdf": helper_cdf.tobytes() == reference_cdf.tobytes(),
                          "cdf_mass": actual["cdf_mass"] == helper_mass == mass == original["cdf_mass"],
                          "uniform": actual["uniform"] == uniform == original["uniform"],
                          "index": actual["selected_index"] == index == original["selected_index"],
                          "source": actual["source"] == source == env.source.tolist(),
                          "positive_support": bool(belief[tuple(source)] > 0),
                          "not_current_cell": source != [26, 26], "one_direct_draw": tape.calls == 1}
                self.compare({"id": f"source-{fixture}-{probe}", **context, "checks": checks,
                              "native_draw": original, "helper_draw": actual,
                              "native_cdf": reference_cdf.tolist(), "helper_cdf": helper_cdf.tolist()})
        draws = {channel: sum(env._public_draw_counts[channel] for env in environments)
                 for channel in ("initial", "source", "hit")}
        self.receipt["native_draws"] = draws
        require(draws == {"initial": 0, "source": 12, "hit": 440}, "complete native draw accounting")
        require(all(self.calls.get(k) == {"attempted": n, "returned": n} for k, n in EXPECTED.items())
                and self.pending is None and len(self.rows) == 461, "exact completed checklist and work")
        require(not any(k in sys.modules for k in ("torch", "tensorflow", "tf_keras")), "no model runtime imported")
        summary = {"version": VERSION, "qualified": all(x["passed"] for x in self.rows), "comparisons": self.rows,
                   "comparison_count": len(self.rows), "failed_comparisons": [x["id"] for x in self.rows if not x["passed"]],
                   "calls": self.calls, "native_draws": draws, "native_restarts": 0, "teacher_choices": 0,
                   "scope": "Fixed one-step generation/transport only; no full teacher panel or efficacy evidence"}
        write(self.out / "summary.json", summary)
        return summary

    def execute(self):
        for path in (self.out, self.args.plan, self.args.supervision):
            regular(path)
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            self.bind()
            result = self.body()
            require(self.plan == metadata() and digest(self.args.plan)["sha256"] == self.args.plan_sha256
                    and digest(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"],
                    "unchanged source/runtime/launch closure")
            require(result["qualified"], "fixed numerical checklist failed; all comparisons retained")
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", qualified=True, calls=self.calls, pending=self.pending,
                                native_draws=result["native_draws"], started_ns=self.start, finished_ns=finished,
                                wall_seconds=(finished-self.start)/1e9, clock_backend=self.clock.backend,
                                files={p.name: digest(p) for p in self.out.iterdir() if p.is_file()})
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "qualified": True,
                              "receipt_sha256": digest(self.out / "receipt.json")["sha256"]}), flush=True)
            return 0
        except BaseException as error:  # Preserve interrupted/pending work and the original error.
            self.receipt.update(status="failed", qualified=False, calls=self.calls, pending=self.pending,
                                operation_stack=self.operation_stack,
                                error=repr(error), traceback=traceback.format_exc())
            try:
                self.receipt["failure_elapsed_seconds"] = (self.clock.now_ns()-self.start)/1e9 if self.clock else None
            except BaseException as secondary:  # noqa: BLE001 - retain the primary failure if timing fails
                self.receipt["failure_clock_error"] = repr(secondary)
            try:
                receipt_path = self.out / "receipt.json"
                if receipt_path.exists():
                    receipt_path.rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: digest(p) for p in self.out.iterdir() if p.is_file()}
                write(receipt_path, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - retain the primary failure if publication fails
                error.add_note(f"Failure receipt publication failed: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    plan = modes.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        regular(args.output)
        write(args.output, metadata())
        print(json.dumps({"plan": str(args.output), **digest(args.output)}), flush=True)
        return 0

    def interrupted(signum, _frame):
        raise InterruptedError(f"qualifier received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return Run(args).execute()
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
