"""Fixed engineering qualification of pinned POBAX RockSample, never training.

Run with the isolated runtime and upstream root on PYTHONPATH. Normal package
imports are intentional: missing or incompatible dependencies are failures.
Injected states test contracts, not achievable policy performance. Five
constructors use the same seed and must produce one identical map.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tmp/pobax-source-review-01"
COMMIT = "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"
PINS = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "pobax/envs/jax/rocksample.py": "88a73c93944b0a9008b726b86a8e2260aecc54b7756a7d805019a726676cfd78",
    "pobax/envs/__init__.py": "eae213739f4ad7b7d8e1afd57de4bf334358055fa07ea0ea813080f44a345749",
    "pobax/envs/wrappers/gymnax.py": "64b7d8c0973ad968292016076d79f06dc35dfe418537c76e3117590a5d65ce83",
    "pobax/envs/configs/rocksample_11_11_config.json": "00b3b1fa07a9115885a8def0e7e7e5be3a9c5bcab46f3e49371f33d8914ce85c",
}
VERSION = "rocksample-runtime-qualification-v1"
MAP_SEED = 260921
LIMITS = {"wall_seconds": 600, "rss_bytes": 8 * 1024**3,
          "output_bytes": 128 * 1024**2, "individual_steps": 2000}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def authenticate_source():
    actual = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    require(actual == COMMIT, "upstream commit changed")
    changes = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "diff", "--name-only", "HEAD"], text=True, timeout=10)
    require(not changes.strip(), "tracked upstream source changed")
    for name, pin in PINS.items():
        require(sha(UPSTREAM / name) == pin, f"upstream pin mismatch: {name}")


class Session:
    def __init__(self, out):
        self.out = out
        self.clock = None
        self.start = None
        self.deadline = None
        self.backend = None
        self.steps_attempted = 0
        self.steps_returned = 0
        self.explicit_reset_calls = 0
        self.records = []
        self.active = "initialization"
        self.supervision = None
        self.imports = {}

    def check(self):
        require(self.clock.now_ns() < self.deadline, "native qualification deadline expired")
        require(rss() <= LIMITS["rss_bytes"], "RSS cap exceeded")
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        require(size <= LIMITS["output_bytes"], "output cap exceeded")
        require(self.steps_attempted <= LIMITS["individual_steps"], "step cap exceeded")

    def record(self, name, status="passed", **details):
        self.active = name
        self.check()
        record = {"name": name, "status": status, **details}
        with (self.out / "checks.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)

    def step(self, name, function, *args, lanes=1):
        self.active = name
        self.check()
        require(self.steps_attempted + lanes <= LIMITS["individual_steps"], "step cap")
        self.steps_attempted += lanes
        result = self.jax.block_until_ready(function(*args))
        require(len(result) == 5, "expected Gymnax five-result step contract")
        self.steps_returned += lanes
        self.check()
        return result

    def reset(self, function, *args):
        self.check()
        self.explicit_reset_calls += 1
        result = self.jax.block_until_ready(function(*args))
        self.check()
        return result

    def timing(self):
        if self.clock is None:
            return {"timing_available": False, "wall_seconds": None}
        try:
            end = self.clock.now_ns()
            return {"timing_available": True, "clock_backend": self.backend,
                    "started_ns": self.start, "finished_ns": end,
                    "deadline_ns": self.deadline, "elapsed_ns": end - self.start,
                    "wall_seconds": (end - self.start) / 1e9}
        except Exception as error:  # noqa: BLE001 - failed timer must not prevent failure preservation
            return {"timing_available": False, "wall_seconds": None, "clock_error": repr(error)}


def initialize(session, args):
    sys.path.insert(0, str(ROOT / "src"))
    from openjev.research.suspend_clock import SuspendClock

    session.clock = SuspendClock()
    session.backend = session.clock.backend
    session.start = session.clock.now_ns()
    session.deadline = session.start + LIMITS["wall_seconds"] * 10**9
    if args.supervision:
        launch_path = Path(args.supervision).resolve()
        wait_until = session.start + 5 * 10**9
        while not launch_path.exists():
            require(session.clock.now_ns() < wait_until, "supervisor launch file missing")
            time.sleep(0.01)
        launch = json.loads(launch_path.read_text())
        command = list(launch["command"])
        if len(command) > 1 and command[1] == "-u":
            command.pop(1)
        require(command == [sys.executable, *sys.argv], "supervisor command mismatch")
        require(launch["version"] == "dialogue-observation-supervision-v2", "supervisor version")
        require(launch["pid"] == os.getpid() == launch["pgid"] == os.getpgrp(), "PID/PGID")
        require(launch["parent_pid"] == os.getppid(), "supervisor parent mismatch")
        require(Path(launch["cwd"]).resolve() == Path.cwd().resolve(), "supervisor cwd")
        require(launch["clock_backend"] == session.backend, "supervisor clock")
        require(launch["cap_seconds"] == LIMITS["wall_seconds"], "supervisor cap")
        require(launch["deadline_ns"] == launch["started_ns"] + 600 * 10**9, "deadline arithmetic")
        require(launch["started_ns"] <= session.start < launch["deadline_ns"], "worker interval")
        session.deadline = launch["deadline_ns"]
        session.supervision = {"path": str(launch_path), "sha256": sha(launch_path),
                               "parent_started_ns": launch["started_ns"]}
    session.check()

    def timeout(_signum, _frame):
        raise TimeoutError("awake emergency alarm expired")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, (session.deadline - session.clock.now_ns()) / 1e9)
    authenticate_source()
    require(importlib.metadata.version("gymnax") == "0.0.9", "requires qualified Gymnax 0.0.9")
    require(importlib.metadata.version("jax") == "0.6.2", "requires resolved JAX 0.6.2")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    write(session.out / "started.json", {
        "version": VERSION, "status": "started", "source_sha256": sha(__file__),
        "upstream_commit": COMMIT, "pinned_sources": PINS, "limits": LIMITS,
        "map_seed": MAP_SEED, "supervision": session.supervision,
        "command": [sys.executable, *sys.argv], "started_civil_unix": time.time(),
        "scope": "Fixed injected engineering states, one unique map, no policy/training/performance."})


def qualify(s):
    # Normal imports exercise the complete installed dependency graph.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gymnax.environments.environment import Environment
    from pobax.envs import get_env, get_transformer_env
    from pobax.envs.jax.rocksample import (
        FullyObservableWrapper,
        PerfectMemoryRockSampleState,
        PerfectMemoryWrapper,
        RockSample,
        RockSampleState,
        half_dist_prob,
    )
    from pobax.envs.wrappers.gymnax import (
        ActionConcatWrapper,
        TimeLimitEnvState,
        TimeLimitWrapper,
        VecEnv,
    )

    s.jax = jax
    require(all(device.platform == "cpu" for device in jax.devices()), "CPU-only qualification")
    require(Path(inspect.getfile(RockSample)).resolve() ==
            UPSTREAM / "pobax/envs/jax/rocksample.py", "imported wrong POBAX checkout")
    s.imports = {"python": sys.version, "devices": [str(d) for d in jax.devices()],
                 "versions": {name: importlib.metadata.version(name) for name in
                              ("jax", "jaxlib", "gymnax", "flax", "chex", "numpy", "navix", "brax")},
                 "gymnax_environment": {"path": inspect.getfile(Environment),
                                        "sha256": sha(inspect.getfile(Environment))},
                 "gymnax_step_source_sha256": hashlib.sha256(
                     inspect.getsource(Environment.step).encode()).hexdigest()}
    imported = {str(Path(module.__file__).resolve().relative_to(UPSTREAM)): sha(module.__file__)
                for module in tuple(sys.modules.values())
                if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(UPSTREAM) + os.sep)
                and str(module.__file__).endswith(".py")}
    s.imports["upstream_imported_source_sha256"] = imported
    write(s.out / "imports.json", s.imports)

    def equal(left, right):
        a, ta = jax.tree_util.tree_flatten(left)
        b, tb = jax.tree_util.tree_flatten(right)
        return ta == tb and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b, strict=True))

    def arr(value):
        return np.asarray(value).tolist()

    def witness(result):
        obs, _state, reward, done, info = result
        return {"observation": arr(obs.obs if hasattr(obs, "obs") else obs),
                "reward": arr(reward), "done": arr(done),
                "info": {k: arr(v) for k, v in info.items()}}

    key = jax.random.PRNGKey
    env = RockSample(key(MAP_SEED), config_path=str(UPSTREAM / "pobax/envs/configs/rocksample_11_11_config.json"))
    params = env.default_params
    coords = np.asarray(env.rock_positions)
    require(coords.shape == (11, 2) and len(set(map(tuple, coords))) == 11, "map geometry")
    require(np.all(coords[:, 1] < 10), "rocks must exclude exit column")
    obs, reset_state = s.reset(env.reset, key(100), params)
    require(obs.shape == (33,) and env.action_space(params).n == 16, "ordinary interface")
    require(np.count_nonzero(np.asarray(obs[22:])) == 0, "reset check channels")
    require(equal((obs, reset_state), s.reset(env.reset, key(100), params)), "reset keyed reproducibility")
    s.record("ordinary_interface_and_keyed_reset", observation=arr(obs), actions=16,
             map_seed=MAP_SEED, rock_positions=arr(coords), reset_quality=arr(reset_state.rock_morality),
             info_scope="Hidden fixture state logged for engineering only; ordinary observations remain 33.")

    good = jnp.zeros(11, dtype=jnp.int32).at[0].set(1)
    bad = jnp.zeros(11, dtype=jnp.int32)
    state = RockSampleState(position=jnp.asarray(coords[0]), rock_morality=good)
    bad_state = RockSampleState(position=state.position, rock_morality=bad)
    good_sample = s.step("sample_good", env.step, key(101), state, 4, params)
    bad_sample = s.step("sample_bad", env.step, key(102), bad_state, 4, params)
    depleted = s.step("sample_depleted", env.step, key(103), good_sample[1], 4, params)
    empty = next((r, c) for r in range(11) for c in range(10) if (r, c) not in set(map(tuple, coords)))
    empty_state = RockSampleState(position=jnp.array(empty), rock_morality=good)
    empty_sample = s.step("sample_empty", env.step, key(104), empty_state, 4, params)
    for name, result, expected in (("good", good_sample, 10), ("bad", bad_sample, -10),
                                   ("depleted", depleted, -10), ("empty", empty_sample, 0)):
        require(float(result[2]) == expected and not bool(result[3]), f"{name} sampling reward")
        require(np.count_nonzero(np.asarray(result[0][22:])) == 0, "sample does not report quality")
        require(equal(result[1].rock_morality, bad if name != "empty" else good), "depletion state")
        s.record(f"sample_{name}", **witness(result))

    for label, fixture, expected in (("good", state, 1), ("bad", bad_state, -1)):
        reading = s.step(f"check_{label}", env.step, key(110), fixture, 5, params)
        require(int(reading[0][22]) == expected and np.count_nonzero(np.asarray(reading[0][22:])) == 1,
                "distance-zero signed reading")
        require(equal(reading[1], fixture) and float(reading[2]) == 0 and not bool(reading[3]), "check physical invariance")
        after = s.step(f"after_check_{label}", env.step, key(111), reading[1], 0, params)
        require(np.count_nonzero(np.asarray(after[0][22:])) == 0, "check reading must disappear")
        s.record(f"ephemeral_check_{label}", check=witness(reading), next_move=witness(after))
    require(float(half_dist_prob(0, 20)) == 1 and float(half_dist_prob(20, 20)) == .75, "sensor formula")
    s.record("sensor_formula", p_at_zero=1., p_at_half_distance=.75,
             scope="Formula plus deterministic distance-zero calls; no empirical stochastic calibration.")

    exit_state = RockSampleState(position=jnp.array([0, 9]), rock_morality=good)
    step_key, reset_key = jax.random.split(key(120))
    direct = s.step("direct_terminal_step_env", env.step_env, step_key, exit_state, 1, params)
    natural = s.step("natural_public_step", env.step, key(120), exit_state, 1, params)
    reset_reference = s.reset(env.reset, reset_key, params)
    require(bool(direct[3]) and float(direct[2]) == 10 and int(direct[1].position[1]) == 10, "direct exit")
    require(bool(natural[3]) and float(natural[2]) == 10, "public exit")
    require(equal(natural[:2], reset_reference), "public natural auto-reset exact keyed reference")
    require(equal(env.rock_positions, coords), "reset must retain constructor map")
    s.record("natural_autoreset_keyed_reference", direct=witness(direct), public=witness(natural),
             reset_reference_observation=arr(reset_reference[0]), map_unchanged=True)

    short = params.replace(max_steps_in_episode=2)
    limited = TimeLimitWrapper(env)
    before_limit = TimeLimitEnvState(env_state=state, elapsed_steps=jnp.array(1))
    truncated = s.step("time_limit_truncation", limited.step, key(130), before_limit, 5, short)
    _inner_key, reset_key = jax.random.split(key(130))
    expected_reset = s.reset(env.reset, reset_key, short)
    require(bool(truncated[3]) and bool(truncated[4]["truncated"]) and bool(truncated[4]["time_limit_reached"]), "truncation flags")
    require(equal(truncated[:2], (expected_reset[0], TimeLimitEnvState(env_state=expected_reset[1], elapsed_steps=jnp.array(0)))), "truncation reset key")
    coincident = s.step("natural_terminal_at_horizon", limited.step, key(131),
                        TimeLimitEnvState(env_state=exit_state, elapsed_steps=jnp.array(1)), 1, short)
    expected_natural = s.step("natural_precedence_reference", env.step, jax.random.split(key(131))[0], exit_state, 1, short)
    require(bool(coincident[3]) and not bool(coincident[4]["truncated"]) and bool(coincident[4]["time_limit_reached"]), "natural termination precedence")
    require(equal(coincident[0], expected_natural[0]) and equal(coincident[1].env_state, expected_natural[1]) and float(coincident[2]) == 10,
            "must keep inner natural reset and reward")
    s.record("time_limit_reset_and_natural_precedence", truncation=witness(truncated), coincident=witness(coincident))

    memory = PerfectMemoryWrapper(env)
    mem_obs, _ = s.reset(memory.reset, key(140), params)
    require(np.count_nonzero(np.asarray(mem_obs[22:])) == 0, "explicit memory reset")
    planted = jnp.zeros(33).at[22:].set(jnp.where(jnp.arange(11) % 2 == 0, 1., -1.))
    mem_exit = PerfectMemoryRockSampleState(position=exit_state.position, rock_morality=good, mem=planted)
    memory_terminal = s.step("legacy_memory_natural_reset", memory.step, key(141), mem_exit, 1, params)
    require(bool(memory_terminal[3]) and equal(memory_terminal[0][22:], planted[22:]), "pinned legacy carry defect changed")
    memory_timed = s.step("legacy_memory_truncation", TimeLimitWrapper(memory).step, key(142),
                          TimeLimitEnvState(env_state=mem_exit, elapsed_steps=jnp.array(1)), 5, short)
    require(bool(memory_timed[4]["truncated"]) and np.count_nonzero(np.asarray(memory_timed[0][22:])) == 0, "outer timeout clears memory")
    s.record("legacy_memory_carries_across_natural_reset", status="known_defect_reproduced",
             natural=witness(memory_terminal), timeout=witness(memory_timed),
             consequence="Carried signs can describe a prior episode after natural reset; explicit/timeout reset clears them.")
    mem_good = PerfectMemoryRockSampleState(position=state.position, rock_morality=good, mem=jnp.zeros(33))
    mem_sample = s.step("legacy_memory_hidden_map_depletion", memory.step, key(143), mem_good, 4, params)
    require(int(mem_sample[0][22]) == -1, "memory wrapper hidden-map depletion witness")
    s.record("legacy_memory_sampling_uses_hidden_map", **witness(mem_sample), privileged_depletion_update=True)

    full = FullyObservableWrapper(env)
    full_obs, full_state = s.reset(full.reset, key(150), params)
    require(equal(full_obs[22:], full_state.rock_morality), "full reset exposes exact hidden quality")
    full_good = s.step("full_observation_good", full.step, key(151), state, 0, params)
    full_bad = s.step("full_observation_bad", full.step, key(151), bad_state, 0, params)
    require(equal(full_good[0][:22], full_bad[0][:22]) and not equal(full_good[0][22:], full_bad[0][22:]), "full hidden-info distinction")
    require(equal(full_good[0][22:], full_good[1].rock_morality), "full quality 0/1 encoding")
    s.record("fully_observable_privileged_qualities", good=witness(full_good), bad=witness(full_bad),
             scope="Quality bits privileged; rock coordinate map still not supplied by this wrapper.")

    action = ActionConcatWrapper(env)
    action_obs, _ = s.reset(action.reset, key(160), params)
    require(action_obs.shape == (49,) and np.count_nonzero(np.asarray(action_obs[33:])) == 0, "initial action suffix")
    action_natural = s.step("action_concat_natural", action.step, key(161), exit_state, 1, params)
    action_timeout = s.step("action_concat_timeout", TimeLimitWrapper(action).step, key(162), before_limit, 5, short)
    require(equal(action_natural[0][33:], jnp.eye(16)[1]), "natural boundary carries terminating action")
    require(np.count_nonzero(np.asarray(action_timeout[0][33:])) == 0, "timeout boundary clears action")
    s.record("action_concat_boundary_asymmetry", natural=witness(action_natural), timeout=witness(action_timeout),
             all_done_suffixes_zero=False)

    # Two lanes: one natural terminal and one continuing, then exact scalar and
    # lane-permutation references. Different keys stay paired with each lane.
    lane_states = [TimeLimitEnvState(env_state=exit_state, elapsed_steps=jnp.array(0)),
                   TimeLimitEnvState(env_state=state, elapsed_steps=jnp.array(0))]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lane_states)
    keys, actions = jax.random.split(key(170), 2), jnp.array([1, 5])
    vector = VecEnv(limited)
    vec = s.step("vector_lane_isolation", vector.step, keys, stacked, actions, params, lanes=2)
    scalar = [s.step(f"scalar_lane_{i}", limited.step, keys[i], lane_states[i], actions[i], params) for i in range(2)]
    reference = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *scalar)
    require(equal(vec, reference), "vector lanes differ from independent scalar calls")
    permuted = s.step("vector_lane_permutation", vector.step, keys[::-1],
                      jax.tree_util.tree_map(lambda x: x[::-1], stacked), actions[::-1], params, lanes=2)
    require(equal(permuted, jax.tree_util.tree_map(lambda x: x[::-1], vec)), "vector permutation changed lanes")
    require(arr(vec[3]) == [True, False], "vector terminal isolation")
    s.record("vector_lane_isolation_and_permutation", **witness(vec))

    def wrapper_chain(wrapped):
        chain = []
        while wrapped is not None:
            chain.append(type(wrapped).__name__)
            wrapped = vars(wrapped).get("_env")
        return chain

    def inject_physical(wrapped_state):
        if hasattr(wrapped_state, "env_state"):
            return dataclasses.replace(wrapped_state, env_state=inject_physical(wrapped_state.env_state))
        return dataclasses.replace(wrapped_state,
                                   position=jnp.stack([exit_state.position, state.position]),
                                   rock_morality=jnp.stack([good, good]))

    routes = []
    for factory in (get_env, get_transformer_env):
        for perfect in (False, True):
            wrapped, factory_params = factory("rocksample_11_11", key(MAP_SEED), num_envs=2,
                                               normalize_env=False, perfect_memory=perfect, action_concat=True)
            require(equal(wrapped.rock_positions, coords), "factory constructed a different map")
            chain = wrapper_chain(wrapped)
            expected = "FullyObservableWrapper" if factory is get_env else "PerfectMemoryWrapper"
            require((expected in chain) == perfect, "factory perfect-memory route mismatch")
            require("TimeLimitWrapper" in chain and "NormalizeVecReward" in chain and "ActionConcatWrapper" in chain, "factory wrappers absent")
            factory_obs, factory_state = s.reset(wrapped.reset, jax.random.split(key(180), 2), factory_params)
            observed = factory_obs.obs if hasattr(factory_obs, "obs") else factory_obs
            require(observed.shape == (2, 49), "factory observation shape")
            result = s.step(f"factory_{factory.__name__}_{perfect}", wrapped.step,
                            jax.random.split(key(181), 2), inject_physical(factory_state), jnp.array([1, 4]), factory_params, lanes=2)
            require(arr(result[3]) == [True, False] and arr(result[4]["reward"]) == [10., 10.], "factory raw reward/done semantics")
            route = {"factory": factory.__name__, "perfect_memory": perfect, "wrapper_chain": chain,
                     "reset_observation": arr(observed), **witness(result),
                     "reward_scope": "Returned rewards normalized; info.reward is original task reward."}
            routes.append(route)
            s.record(f"factory_{factory.__name__}_{perfect}", **route)
    authenticate_source()
    for name, pin in imported.items():
        require(sha(UPSTREAM / name) == pin, f"imported source changed: {name}")
    return {"map_seed": MAP_SEED, "unique_maps": 1, "map_constructors": 5,
            "factory_routes": routes, "ordinary_observation_size": 33, "actions": 16,
            "action_concat_observation_size": 49, "known_defects_reproduced": [
                "PerfectMemoryWrapper carries old signed readings across natural auto-reset"],
            "boundary_asymmetries": ["Natural done retains action one-hot; time-limit truncation clears it"],
            "scope": "Engineering states and exact keyed references only; no return benchmark, learning, policy, map generalization or sensor-frequency claim."}


def execute(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    session = Session(out)
    source_pin = sha(__file__)
    try:
        initialize(session, args)
        summary = qualify(session)
        summary.update(version=VERSION, status="completed", checks=len(session.records),
                       steps_attempted=session.steps_attempted, steps_returned=session.steps_returned,
                       explicit_reset_calls=session.explicit_reset_calls, model_calls=0, training_updates=0,
                       limits=LIMITS, imports=session.imports)
        require(session.steps_attempted == session.steps_returned, "unreturned environment step")
        write(out / "summary.json", summary)
        session.check()
        require(sha(__file__) == source_pin, "qualification source changed")
        if session.supervision:
            require(sha(session.supervision["path"]) == session.supervision["sha256"], "launch changed")
        files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(out.iterdir()) if p.is_file()}
        timing = session.timing()
        require(timing["timing_available"] and timing["finished_ns"] < session.deadline, "late completion")
        write(out / "completed.json", {"version": VERSION, "status": "completed",
              "source_sha256": source_pin, "upstream_commit": COMMIT, "files": files,
              "supervision": session.supervision, "checks": len(session.records),
              "steps_attempted": session.steps_attempted, "steps_returned": session.steps_returned,
              "model_calls": 0, "training_updates": 0, "peak_rss_bytes": rss(), **timing})
        session.check()
        print(json.dumps({"status": "completed", "checks": len(session.records),
                          "steps": session.steps_returned, "completed_sha256": sha(out / "completed.json")}))
    except BaseException as error:
        # An unrelated failure near the deadline must keep its original cause.
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "late-completed.json")
            write(out / "failed.json", {"version": VERSION, "status": "failed", "error": repr(error),
                  "active_check": session.active, "source_sha256": source_pin,
                  "supervision": session.supervision, "checks_returned": len(session.records),
                  "steps_attempted": session.steps_attempted, "steps_returned": session.steps_returned,
                  "peak_rss_bytes": rss(), "training_updates": 0, "model_calls": 0, **session.timing()})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original exception
            if hasattr(error, "add_note"):
                error.add_note(f"Failure preservation also failed: {secondary!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--supervision")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
