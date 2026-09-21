"""Prospective autonomous public-input spectral control, two fixed OTTO regimes.

Native imports happen only after source/runtime/supervisor authentication. No
training, rank selection, inherited gate revision or architectural novelty claim.
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
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tmp/otto-source-review-01"
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

VERSION = "otto-spectral-control-v1"
ARMS = ("full_bayes", "exact_log", "recent32", "recent32_hard", "dct16_neutral", "dct16_nearest")
CANDIDATES = ARMS[-2:]
CASES, HORIZON, QUALIFICATION_STEP_CAP = 96, 2188, 2048
LIMITS = {"native_seconds": 900, "rss_bytes": 4 * 1024**3, "output_bytes": 512 * 1024**2,
          "native_steps": 2522624, "qualification_steps": QUALIFICATION_STEP_CAP}
COHORTS = {name: {"first_seed": seed, "qualification_seed": qual,
                  "config": {"Ndim": 2, "lambda_over_dx": lam, "R_dt": 2.0,
                             "norm_Poisson": "Euclidean", "Ngrid": 53, "Nhits": 4}}
           for name, seed, qual, lam in (("base", 630001, 650001, 3.0), ("shift", 640001, 660001, 4.0))}
CONFIGURATION = {"arms": list(ARMS), "cohorts": COHORTS, "cases_per_cohort": CASES,
                 "horizon": HORIZON, "episodes": 1152, "rotation": "global_case_index modulo6",
                 "score_tolerance": 1e-8, "posterior_tv_tolerance": 1e-10,
                 "learned_pilot_admission": False}
PINNED = {
    "scripts/study_otto_spectral_memory.py": "9bae9a6095061fc7f6c203ec34bdf19d174e7d13822e348231b98c8884447f88",
    "scripts/audit_otto_large_memory.py": "1ad5a080f58177751dadf0eaab0cd9e2c7b37d7e9d84b50cc32f4046a66155ec",
    "src/openjev/research/otto_spectral_memory.py": "440c7527a6040fb9b41985dc8f48e5f62e18be880242de09f70b46b540579943",
    "src/openjev/research/otto_public.py": "438631a18005493e0cafa0777315e2158cac97cc7d0e71ad9fd6402284615b3d",
    "src/openjev/research/suspend_clock.py": "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
    "scripts/supervise_dialogue_observation_v2.py": "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144",
}
NEW_SOURCES = {"scripts/study_otto_spectral_control.py", "tests/test_otto_spectral_control.py",
               "research/otto-spectral-control-protocol.md"}
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
TIMES = ("actor_initialization_seconds", "update_seconds", "decode_seconds", "planner_seconds",
         "shared_initialization_allocation_seconds")
METRICS = ("capped_time", "found", "stuck_steps", *TIMES, "controller_seconds",
           "environment_initialization_seconds", "environment_seconds", "episode_seconds", "state_array_bytes")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def emit(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def public(packet):
    require(set(packet._fields) == {"position", "hit", "done", "step", "valid_actions"}, "public input whitelist")
    return {"position": list(packet.position), "hit": packet.hit, "done": packet.done,
            "step": packet.step, "valid_actions": list(packet.valid_actions)}


def case_order():
    for cohort_index, (name, cfg) in enumerate(COHORTS.items()):
        for case in range(CASES):
            shift = (cohort_index * CASES + case) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                yield name, cfg["first_seed"] + case, case // 12, 1 + (case % 12) // 4, arm


def evolving_bytes(actor):
    record = actor.storage_bytes()
    return record.get("state_array_bytes", record.get("mutable_array_bytes"))


def criteria(means, blocks, candidate):
    c, full, exact = (means[a] for a in (candidate, "full_bayes", "exact_log"))
    checks = []

    def add(name, value, threshold, relation):
        passes = value >= threshold if relation == ">=" else value <= threshold if relation == "<=" else value < threshold
        checks.append({"name": name, "value": value, "threshold": threshold, "relation": relation, "passes": bool(passes)})

    add("full_success_at_least_95pct", full["found"], .95, ">=")
    for name, baseline in (("full_bayes", full), ("exact_log", exact)):
        add(f"no_failure_regression_vs_{name}", c["found"] - baseline["found"], 0, ">=")
        add(f"time_at_most_105pct_{name}", c["capped_time"], 1.05 * baseline["capped_time"], "<=")
    for name in ("recent32", "recent32_hard"):
        add(f"strict_time_gain_vs_{name}", c["capped_time"], means[name]["capped_time"], "<")
    gains = [b["means"]["recent32_hard"]["capped_time"] - b["means"][candidate]["capped_time"] for b in blocks]
    add("positive_blocks_vs_recent32_hard_at_least_six", sum(g > 0 for g in gains), 6, ">=")
    checks[-1]["block_gains"] = gains
    add("state_at_most_20pct_full", c["state_array_bytes"], .20 * full["state_array_bytes"], "<=")
    add("controller_at_most_150pct_full", c["controller_seconds"], 1.5 * full["controller_seconds"], "<=")
    utility = [
        {"name": "no_failure_regression_vs_full", "passes": c["found"] >= full["found"], "value": c["found"] - full["found"]},
        {"name": "time_no_worse_than_full", "passes": c["capped_time"] <= full["capped_time"], "value": c["capped_time"] - full["capped_time"]},
        {"name": "controller_no_worse_than_full", "passes": c["controller_seconds"] <= full["controller_seconds"], "value": c["controller_seconds"] - full["controller_seconds"]},
        {"name": "time_or_controller_strictly_better", "passes": c["capped_time"] < full["capped_time"] or c["controller_seconds"] < full["controller_seconds"]},
    ]
    return checks, utility


def summarize(rows, weights, *, qualification):
    expected = list(case_order())
    require(len(rows) == len(expected) and [(r["cohort"], r["seed"], r["block"], r["initial_hit"], r["arm"]) for r in rows] == expected,
            "complete exact rotated two-cohort membership")
    require(qualification["status"] == "completed" and qualification["checks"]
            and all(c["passes"] is True for c in qualification["checks"])
            and 0 < qualification["native_steps"] <= QUALIFICATION_STEP_CAP, "qualification before complete scoring")
    require(set(weights) == set(COHORTS), "both mixtures")
    for r in rows:
        require(type(r["steps"]) is int and 1 <= r["steps"] <= HORIZON and r["capped_time"] == r["steps"]
                and type(r["found"]) is bool and (r["found"] or r["steps"] == HORIZON), "capped outcome semantics")
        require(type(r["stuck_steps"]) is int and 0 <= r["stuck_steps"] <= r["steps"], "stuck step count")
        require(all(type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] >= 0 for k in METRICS if k != "found"), "finite metrics")
        require(abs(r["controller_seconds"] - math.fsum(r[k] for k in TIMES)) <= 1e-9, "controller accounting")
        require(r["update_calls"] == r["steps"] - int(r["found"])
                and r["decode_calls"] == r["planner_calls"] == r["steps"], "complete actor operation counts")
    result = {}
    for name in COHORTS:
        w = weights[name]
        require(set(w) == {1, 2, 3} and all(math.isfinite(v) and 0 < v < 1 for v in w.values())
                and abs(math.fsum(w.values()) - 1) <= 1e-12, "cohort-specific positive mixture")
        selected = [r for r in rows if r["cohort"] == name]

        def group(subset, mixture=w):
            return {arm: {m: math.fsum(mixture[h] * math.fsum(float(r[m]) for r in subset if r["arm"] == arm and r["initial_hit"] == h)
                                      / sum(r["arm"] == arm and r["initial_hit"] == h for r in subset) for h in (1, 2, 3))
                          for m in METRICS} for arm in ARMS}

        means = group(selected)
        strata = {h: {arm: {m: math.fsum(float(r[m]) for r in selected if r["arm"] == arm and r["initial_hit"] == h) / 32
                           for m in METRICS} for arm in ARMS} for h in (1, 2, 3)}
        blocks = [{"block": b, "means": group([r for r in selected if r["block"] == b])} for b in range(8)]
        decisions = {a: criteria(means, blocks, a) for a in CANDIDATES}
        result[name] = {"initial_hit_weights": w, "means": means, "strata": strata, "blocks": blocks,
                        "criteria_by_candidate": {a: decisions[a][0] for a in CANDIDATES},
                        "utility_criteria_by_candidate": {a: decisions[a][1] for a in CANDIDATES},
                        "compact_by_candidate": {a: all(c["passes"] for c in decisions[a][0]) for a in CANDIDATES},
                        "utility_by_candidate": {a: all(c["passes"] for c in decisions[a][1]) for a in CANDIDATES}}
    return {"scope": "Autonomous untrained control on two fresh fixed-grid regimes; no learned admission or original gate revision.",
            "episodes": len(rows), "cohorts": result, "structural_qualification": True,
            "compact_control_viable": all(result[n]["compact_by_candidate"][a] for n in COHORTS for a in CANDIDATES),
            "utility_compute_advantage": all(result[n]["utility_by_candidate"][a] for n in COHORTS for a in CANDIDATES),
            "utility_by_candidate": {a: all(result[n]["utility_by_candidate"][a] for n in COHORTS) for a in CANDIDATES},
            "learned_pilot_admission": False, "inherited_gate_revised": False}


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.clock = self.start = self.launch = None
        self.receipt = {"status": "started", "native_steps_attempted": 0, "native_steps_returned": 0,
                        "completed_episodes": 0, "training_updates": 0, "external_model_calls": 0, "limits": LIMITS}

    def check(self):
        if self.launch is not None:
            require(self.clock.now_ns() < self.launch["deadline_ns"], "native shared deadline expired")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "RSS cap")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "output cap")
        require(self.receipt["native_steps_attempted"] <= LIMITS["native_steps"], "total native step cap")

    def step(self, env, action, hit=None):
        self.check()
        require(self.receipt["native_steps_attempted"] < LIMITS["native_steps"], "native allocation exhausted")
        if self.receipt.get("phase") == "qualification":
            require(self.receipt["native_steps_attempted"] < QUALIFICATION_STEP_CAP, "qualification allocation exhausted")
        self.receipt["native_steps_attempted"] += 1
        result = env.step(action, hit=hit, quiet=True)
        self.receipt["native_steps_returned"] += 1
        return result

    def authenticate(self):
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_native_run"
                and self.plan["configuration"] == CONFIGURATION and self.plan["limits"] == LIMITS, "frozen study identity")
        require(NEW_SOURCES | PINNED.keys() <= self.plan["sources"].keys(), "required source closure")
        for name, pin in PINNED.items():
            require(self.plan["sources"][name] == pin, f"fixed inherited source: {name}")
        for name, pin in self.plan["sources"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and not (ROOT / name).is_symlink()
                    and sha(ROOT / name) == pin, f"source identity: {name}")
        require(subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
                == self.plan["upstream_commit"], "upstream revision")
        require(not subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "HEAD", "--name-only"], text=True).strip(), "modified upstream")
        for name, pin in self.plan["upstream_sources"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and not (UPSTREAM / name).is_symlink()
                    and sha(UPSTREAM / name) == pin, f"upstream identity: {name}")
        extras = subprocess.check_output(["git", "-C", str(UPSTREAM), "ls-files", "--others", "--exclude-standard"], text=True)
        require(not any(p.endswith(".py") for p in extras.splitlines()), "untracked upstream Python")
        for name, version in self.plan["runtime_versions"].items():
            require(importlib.metadata.version(name) == version, f"runtime: {name}")
        require(sys.version.split()[0] == self.plan["python_version"] and str(Path(sys.executable).absolute()) == self.plan["python_executable"], "Python identity")

    def bind(self):
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "supervision missing")
            time.sleep(.01)
        self.launch = json.loads(self.args.supervision.read_text())
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                and self.launch["cap_seconds"] == LIMITS["native_seconds"] and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["native_seconds"] * 10**9
                and Path(self.launch["cwd"]).resolve() == ROOT == Path.cwd().resolve()
                and self.launch["watchdog_sha256"] == PINNED["scripts/supervise_dialogue_observation_v2.py"]
                and self.launch["clock_source_sha256"] == PINNED["src/openjev/research/suspend_clock.py"], "supervisor identity")
        require(sha(self.args.plan) == self.args.plan_sha256, "external plan pin")
        self.plan = json.loads(self.args.plan.read_text())
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision),
                            sources=self.plan["sources"], upstream_sources=self.plan["upstream_sources"])
        self.authenticate()
        self.check()

    def qualify(self, SourceTracking, HeuristicPolicy, seeded_environment, observation, saved, spectral, analytic, np):
        from scipy.special import k0
        from scipy.stats import poisson

        checks, weights, models, kernels, shared_times, template_times = [], {}, {}, {}, {}, {}
        with (self.out / "qualification.jsonl").open("x") as journal:
            def record(name, passed, **details):
                item = {"name": name, "passes": bool(passed), **details}
                checks.append(item)
                emit(journal, item)
                journal.flush()
                require(passed, f"qualification: {name}")

            for name, cfg in COHORTS.items():
                self.check()
                config, qseed = cfg["config"], cfg["qualification_seed"]
                tick = time.perf_counter()
                template = SourceTracking(**config, draw_source=False, initial_hit=1)
                template_times[name] = time.perf_counter() - tick
                record(f"{name}.explicit_geometry", template.N == 53 and template.Nhits == 4)
                lam = config["lambda_over_dx"]
                cells = np.indices((107, 107)) - 53
                d = np.sqrt(np.sum(cells**2, axis=0))
                mu = 2 * k0(np.where(d == 0, 1, d) / lam) / np.log(2 * lam)
                theory = np.stack((poisson.pmf(0, mu), poisson.pmf(1, mu), poisson.pmf(2, mu), poisson.sf(2, mu)))
                theory[:, 53, 53] = 0
                kernel = template.p_Poisson.copy()
                error = float(np.max(np.abs(kernel - theory)))
                record(f"{name}.kernel_formula", error <= 1e-12, maximum_absolute_difference=error,
                       zero_counts=[int((k == 0).sum()) for k in kernel])
                radius = np.arange(1, int(1000 * lam))
                means = 2 * k0(radius / lam) / np.log(2 * lam)
                shell = np.pi * ((radius + .5)**2 - (radius - .5)**2)
                masses = np.array([0, *(float(np.sum(poisson.pmf(h, means) * shell)) for h in (1, 2)),
                                   float(np.sum(poisson.sf(2, means) * shell))])
                expected = masses / masses.sum()
                initial_env = seeded_environment(SourceTracking, qseed, config)
                draw = next(d for d in initial_env.draw_log if d["channel"] == "initial")
                actual = np.asarray(draw["probabilities"])
                error = float(np.max(np.abs(actual - expected)))
                record(f"{name}.initial_hit_mixture", error <= 1e-12 and actual[0] == 0 and np.all(actual[1:] > 0),
                       probabilities=actual.tolist(), maximum_absolute_difference=error)
                weights[name] = {h: float(actual[h]) for h in (1, 2, 3)}
                kernel.setflags(write=False)
                tick = time.perf_counter()
                model = spectral.SpectralModel(kernel)
                shared_times[name] = time.perf_counter() - tick
                kernels[name], models[name] = kernel, model
                np.savez_compressed(self.out / f"public-kernel-{name}.npz", likelihood=kernel, initial_hit_weights=actual)

                def forced(source, hit, configuration=config):
                    class Forced(SourceTracking):
                        def _draw_a_source(self):
                            self.source = source.copy()
                    return Forced(**configuration, draw_source=True, initial_hit=hit)

                for hit in (1, 2, 3):
                    state_before = np.random.get_state()
                    first = seeded_environment(SourceTracking, qseed + 9 + hit, config, initial_hit=hit)
                    second = seeded_environment(SourceTracking, qseed + 9 + hit, config, initial_hit=hit)
                    reference = forced(first.source, hit)
                    packet = public(observation(first, 0))
                    actors = {a: saved.make_actor(a, packet, model, kernel, analytic, np)
                              for a in ("full_bayes", "exact_log", "dct53_neutral", "dct53_nearest")}
                    native_policy = HeuristicPolicy(first, policy=1)
                    max_tv = max_score = 0.0
                    replay = []
                    require(np.array_equal(first.source, second.source) and np.array_equal(first.p_source, second.p_source), "paired initialization")
                    path = [0] * 8 + [2] * 8
                    for step in range(1, 85):
                        native_action, raw_scores = native_policy._space_aware_infotaxis()
                        for actor in actors.values():
                            logs, p = actor.decode()
                            saved.validate_decoded(np, logs, p, (53, 53))
                            tv = float(np.abs(p - first.p_source).sum() / 2)
                            scores = analytic.action_scores(p, tuple(packet["position"]), kernel, True)
                            score_error = max(abs(scores[a] - float(raw_scores[a])) for a in packet["valid_actions"])
                            require(tv <= 1e-10 and score_error <= 1e-8 and saved.choose(scores)[0] == int(native_action), "native full/exact/fullrank parity")
                            max_tv, max_score = max(max_tv, tv), max(max_score, score_error)
                        if path:
                            action = path.pop(0)
                        else:
                            differences = first.source - first.agent
                            axis = int(np.flatnonzero(differences)[0])
                            action = 2 * axis + int(differences[axis] > 0)
                        result = self.step(first, action)
                        clone = self.step(second, action)
                        replayed = self.step(reference, action, hit=int(result[0]))
                        require(result == clone == replayed and np.array_equal(first.p_source, reference.p_source)
                                and np.array_equal(first.p_source, second.p_source), "seeded forced-hit replay")
                        packet = public(observation(first, step))
                        replay.append({"step": step, "action": action, "public": packet})
                        if packet["done"]:
                            break
                        for actor in actors.values():
                            actor.update(packet)
                    after = np.random.get_state()
                    record(f"{name}.replay_hit{hit}", packet["done"] and first.draw_log == second.draw_log,
                           source_evaluation_only=first.source.tolist(), steps=step, replay=replay,
                           draw_log=first.draw_log, maximum_tv=max_tv, maximum_score_error=max_score)
                    record(f"{name}.global_rng_hit{hit}", state_before[0] == after[0]
                           and np.array_equal(state_before[1], after[1]) and state_before[2:] == after[2:])
                fixture = forced(np.array([52, 52]), 1)
                result = self.step(fixture, 0, hit=3)
                for _ in range(30):
                    self.step(fixture, 0, hit=0)
                position = tuple(fixture.agent)
                self.step(fixture, 0, hit=0)
                boundary = tuple(fixture.agent) == position and position[0] == 0
                count = 32
                while not fixture.obs["done"]:
                    axis = int(np.flatnonzero(fixture.source - fixture.agent)[0])
                    self.step(fixture, 2 * axis + int(fixture.source[axis] > fixture.agent[axis]), hit=0)
                    count += 1
                record(f"{name}.saturated_boundary_found", result == (3, 0, False) and boundary and count == 110
                       and fixture.obs == {"hit": -2, "done": True}, native_steps=count)
        qualified = {"status": "completed", "checks": checks, "native_steps": self.receipt["native_steps_returned"],
                     "initial_hit_weights": weights, "shared_model_initialization_seconds": shared_times,
                     "template_initialization_seconds": template_times,
                     "scope": "Injected paths qualify implementation, not policy utility; no cohort outcome selection."}
        require(qualified["native_steps"] <= QUALIFICATION_STEP_CAP, "bounded qualification")
        write(self.out / "qualification.json", qualified)
        return qualified, models, kernels

    def cohort(self, SourceTracking, seeded_environment, observation, saved, analytic, np, qualified, models, kernels):
        rows, maximum_parity = [], 0.0
        with (self.out / "transitions.jsonl").open("x") as transitions, (self.out / "episodes.jsonl").open("x") as episodes:
            for name, seed, block, hit, arm in case_order():
                self.check()
                episode_start = time.perf_counter()
                self.receipt["active_case"] = {"cohort": name, "seed": seed, "arm": arm}
                tick = time.perf_counter()
                env = seeded_environment(SourceTracking, seed, COHORTS[name]["config"], initial_hit=hit)
                environment_init = time.perf_counter() - tick
                tick = time.perf_counter()
                packet = public(observation(env, 0))
                actor = saved.make_actor(arm, packet, models[name], kernels[name], analytic, np)
                times = {"actor_initialization_seconds": time.perf_counter() - tick,
                         "update_seconds": 0.0, "decode_seconds": 0.0, "planner_seconds": 0.0, "environment_seconds": 0.0,
                         "shared_initialization_allocation_seconds": qualified["shared_model_initialization_seconds"][name] / CASES
                         if arm == "exact_log" or arm in CANDIDATES else 0.0}
                state_bytes, storage = evolving_bytes(actor), actor.storage_bytes()
                stuck = updates = 0
                emit(transitions, {"kind": "reset", "cohort": name, "seed": seed, "arm": arm, "block": block,
                                   "initial_hit": hit, "source_evaluation_only": env.source.tolist(), "public": packet})
                for step in range(1, HORIZON + 1):
                    self.check()
                    tick = time.perf_counter()
                    logs, probabilities = actor.decode()
                    decoding = time.perf_counter() - tick
                    saved.validate_decoded(np, logs, probabilities, (53, 53))
                    error = None
                    if arm in ("full_bayes", "exact_log"):
                        error = float(np.abs(probabilities - env.p_source).sum() / 2)
                        maximum_parity = max(maximum_parity, error)
                        require(error <= 1e-10, "cohort full/exact posterior parity")
                    tick = time.perf_counter()
                    scores = analytic.action_scores(probabilities, tuple(packet["position"]), kernels[name], True)
                    action = saved.choose(scores)[0]
                    planning = time.perf_counter() - tick
                    require(action in packet["valid_actions"], "valid chosen action")
                    self.receipt["active_transition"] = {"cohort": name, "seed": seed, "arm": arm, "step": step,
                                                         "action": action, "scores": scores, "public_before": packet}
                    tick = time.perf_counter()
                    self.step(env, action)
                    environment = time.perf_counter() - tick
                    after = public(observation(env, step))
                    self.receipt["active_transition"]["public_after"] = after
                    updating = 0.0
                    if not after["done"]:
                        tick = time.perf_counter()
                        actor.update(after)
                        updating = time.perf_counter() - tick
                        updates += 1
                    for key, value in (("update_seconds", updating), ("decode_seconds", decoding),
                                       ("planner_seconds", planning), ("environment_seconds", environment)):
                        times[key] += value
                    stuck += int(env.agent_stuck)
                    emit(transitions, {"kind": "step", "cohort": name, "seed": seed, "arm": arm, "step": step,
                                       "action": action, "scores": scores, "public": after, "found": after["done"],
                                       "stuck": bool(env.agent_stuck), "posterior_mass": float(probabilities.sum()),
                                       "posterior_before_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(),
                                       "full_exact_tv_before": error, "update_seconds": updating, "decode_seconds": decoding,
                                       "planner_seconds": planning, "environment_seconds": environment})
                    packet = after
                    if packet["done"]:
                        break
                require(actor.storage_bytes() == storage, "fixed per-actor array allocation")
                row = {"cohort": name, "seed": seed, "arm": arm, "block": block, "initial_hit": hit,
                       "steps": step, "capped_time": step, "found": packet["done"], "stuck_steps": stuck,
                       **times, "environment_initialization_seconds": environment_init,
                       "episode_seconds": time.perf_counter() - episode_start,
                       "controller_seconds": math.fsum(times[k] for k in TIMES), "state_array_bytes": state_bytes,
                       "storage": storage, "update_calls": updates, "decode_calls": step, "planner_calls": step,
                       "source_evaluation_only": env.source.tolist(), "draw_log": env.draw_log}
                rows.append(row)
                emit(episodes, row)
                episodes.flush()
                transitions.flush()
                self.receipt["completed_episodes"] = len(rows)
                self.receipt.pop("active_transition", None)
                print(json.dumps({"completed_episodes": len(rows), "native_steps": self.receipt["native_steps_returned"]}), flush=True)
        weights = qualified["initial_hit_weights"]
        summary = summarize(rows, weights, qualification=qualified)
        summary.update(full_exact_maximum_tv=maximum_parity,
                       shared_model={name: model.storage_bytes() for name, model in models.items()},
                       shared_model_initialization_seconds=qualified["shared_model_initialization_seconds"],
                       sum_controller_seconds=math.fsum(r["controller_seconds"] for r in rows),
                       sum_measured_actor_seconds=math.fsum(r["controller_seconds"] - r["shared_initialization_allocation_seconds"] for r in rows),
                       sum_environment_seconds=math.fsum(r["environment_seconds"] + r["environment_initialization_seconds"] for r in rows),
                       timing_scope="One rotated autonomous pass; controller includes initialization, all updates, decodes, planning and one regime-specific shared-model setup/96 allocation for each exact/DCT episode. Shared model was physically constructed once per regime, so allocated controller costs are not a disjoint whole-process sum. Whole episode excludes its own final record serialization; environment, qualification, validation, serialization and process overhead are separately reported or contained in supervisor elapsed.")
        write(self.out / "summary.json", summary)
        self.receipt.update(compact_control_viable=summary["compact_control_viable"],
                            utility_compute_advantage=summary["utility_compute_advantage"], learned_pilot_admission=False,
                            full_exact_maximum_tv=maximum_parity)

    def execute(self):
        primary = None
        try:
            self.clock, self.start = SuspendClock(), None
            self.start = self.clock.now_ns()
            self.bind()
            write(self.out / "started.json", {"request": {k: str(v) for k, v in vars(self.args).items()},
                  "clock_backend": self.clock.backend, "started_ns": self.start,
                  "deadline_ns": self.launch["deadline_ns"], "supervision_sha256": self.receipt["supervision_sha256"]})
            for name, value in THREAD_ENV.items():
                os.environ[name] = value
            import numpy as np
            sys.path.insert(0, str(UPSTREAM))
            from isotropic.classes.heuristicpolicy import HeuristicPolicy
            from isotropic.classes.sourcetracking import SourceTracking

            from openjev.research.otto_public import observation, seeded_environment
            saved = load("scripts/study_otto_spectral_memory.py", "spectral_control_saved")
            spectral = load("src/openjev/research/otto_spectral_memory.py", "spectral_control_model")
            analytic = load("scripts/audit_otto_large_memory.py", "spectral_control_math")
            for name in ("sourcetracking", "heuristicpolicy", "policy"):
                require(Path(sys.modules[f"isotropic.classes.{name}"].__file__).resolve()
                        == UPSTREAM / f"isotropic/classes/{name}.py", "normal pinned upstream imports")
            write(self.out / "imports.json", {"python": sys.version, "executable": sys.executable,
                  "versions": {name: importlib.metadata.version(name) for name in self.plan["runtime_versions"]},
                  "numerical_thread_environment": THREAD_ENV,
                  "upstream": {name: str(sys.modules[name].__file__) for name in
                               ("isotropic.classes.sourcetracking", "isotropic.classes.heuristicpolicy", "isotropic.classes.policy")}})
            self.receipt["phase"] = "qualification"
            qualified, models, kernels = self.qualify(SourceTracking, HeuristicPolicy, seeded_environment, observation, saved, spectral, analytic, np)
            self.receipt["qualification_steps"] = self.receipt["native_steps_returned"]
            self.receipt["phase"] = "cohort"
            self.cohort(SourceTracking, seeded_environment, observation, saved, analytic, np, qualified, models, kernels)
            require(self.receipt["native_steps_attempted"] == self.receipt["native_steps_returned"]
                    and self.receipt["completed_episodes"] == 1152, "all native calls and episodes returned")
            self.authenticate()
            require(sha(self.args.plan) == self.args.plan_sha256
                    and sha(self.args.supervision) == self.receipt["supervision_sha256"], "unchanged plan and live launch")
            self.check()
            self.receipt["status"] = "completed"
        except BaseException as error:
            primary = error
            self.receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            failures = []
            try:
                self.receipt.update(clock_backend=self.clock.backend, started_ns=self.start,
                                    finished_ns=self.clock.now_ns())
                self.receipt["elapsed_ns"] = self.receipt["finished_ns"] - self.start
                self.receipt["native_elapsed_seconds"] = self.receipt["elapsed_ns"] / 1e9
            except BaseException as error:  # noqa: BLE001 - keep original failure.
                self.receipt.update(finished_ns=None, elapsed_ns=None, native_elapsed_seconds=None)
                failures.append(f"clock finalization: {error!r}")
            self.receipt["timing_scope"] = "Worker before receipt publication; supervisor includes process lifetime. Phase timings are perf_counter diagnostics."
            try:
                self.receipt["files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in self.out.iterdir() if p.is_file()}
                if failures:
                    self.receipt.update(status="failed", finalization_errors=failures)
                write(self.out / "receipt.json", self.receipt)
            except BaseException as error:  # noqa: BLE001 - keep original failure.
                failures.append(f"receipt finalization: {error!r}")
                print(json.dumps({"status": "failed", "original_error": repr(primary), "finalization_errors": failures}), file=sys.stderr)
            if failures and primary is None:
                raise RuntimeError("; ".join(failures))
        try:
            self.check()
        except BaseException as error:
            self.receipt.update(status="failed", error=f"late publication: {error!r}")
            try:
                write(self.out / "late-failure.json", self.receipt)
            except OSError as secondary:
                error.add_note(f"Late failure publication: {secondary!r}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("plan", "supervision", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    Run(parser.parse_args()).execute()
