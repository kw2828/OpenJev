"""Qualify seeded OTTO sampled-source search, then run a frozen memory screen."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tmp/otto-source-review-01"
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

ARMS = ("space_full", "space_recent32", "space_recent8", "space_initial",
        "info_full", "info_recent32", "info_recent8", "info_initial")
HORIZON = 642
CONFIG = {"Ndim": 2, "lambda_over_dx": 1.0, "R_dt": 1.0,
          "norm_Poisson": "Euclidean", "Ngrid": None, "Nhits": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def require(value, message):
    if not value:
        raise AssertionError(message)


def summarize(rows, weights, *, qualification):
    structural = (qualification.get("status") == "completed"
                  and bool(qualification.get("checks"))
                  and all(c.get("passes") is True for c in qualification["checks"])
                  and type(qualification.get("native_steps")) is int
                  and 0 < qualification["native_steps"] <= 1000)
    require(set(weights) == {1, 2} and all(math.isfinite(w) and 0 < w < 1 for w in weights.values())
            and abs(sum(weights.values()) - 1) <= 1e-12, "invalid initial-hit mixture")
    expected = {(seed, arm) for seed in range(510001, 510065) for arm in ARMS}
    require(len(rows) == 512 and {(r["seed"], r["arm"]) for r in rows} == expected,
            "incomplete or duplicated cohort")
    for r in rows:
        case = r["seed"] - 510001
        require(r["block"] == case // 8 and r["initial_hit"] == 1 + (case % 8) // 4,
                "stratum or block assignment changed")
        require(type(r["steps"]) is int and 1 <= r["steps"] <= HORIZON
                and r["capped_time"] == r["steps"] and type(r["found"]) is bool, "invalid capped time")
        require(r["found"] or r["steps"] == HORIZON, "early unsuccessful termination")
        require(type(r["stuck_steps"]) is int and 0 <= r["stuck_steps"] <= r["steps"], "invalid stuck count")
        times = ("controller_seconds", "initialization_seconds", "filter_seconds", "planning_seconds", "environment_seconds")
        require(all(math.isfinite(r[k]) and r[k] >= 0 for k in times), "invalid measured time")
        require(abs(r["controller_seconds"] - sum(r[k] for k in times[1:4])) <= 1e-9,
                "controller timing sum disagrees")
    metrics = ("capped_time", "found", "controller_seconds", "initialization_seconds",
               "filter_seconds", "planning_seconds", "environment_seconds", "stuck_steps")

    def group(subset):
        means = {}
        for arm in ARMS:
            strata = {h: [r for r in subset if r["arm"] == arm and r["initial_hit"] == h] for h in (1, 2)}
            require(all(strata.values()), "missing initial-hit stratum")
            means[arm] = {m: sum(weights[h] * sum(r[m] for r in strata[h]) / len(strata[h]) for h in (1, 2))
                          for m in metrics}
        return means

    all_means = group(rows)
    strata = {h: {arm: {m: sum(r[m] for r in rows if r["initial_hit"] == h and r["arm"] == arm) / 32
                        for m in metrics} for arm in ARMS} for h in (1, 2)}
    blocks = [{"block": b, "means": group([r for r in rows if r["block"] == b])} for b in range(8)]
    full = all_means["space_full"]
    recent = all_means["space_recent32"]
    gain = recent["capped_time"] - full["capped_time"]
    block_gains = [b["means"]["space_recent32"]["capped_time"] - b["means"]["space_full"]["capped_time"] for b in blocks]
    criteria = [
        {"name": "full_success_at_least_95pct", "value": full["found"], "passes": full["found"] >= .95},
        {"name": "gain_vs_recent32_at_least_10pct", "value": gain / recent["capped_time"], "passes": gain / recent["capped_time"] >= .10},
        {"name": "gain_vs_recent32_at_least_two_steps", "value": gain, "passes": gain >= 2},
        {"name": "positive_blocks_vs_recent32_at_least_six", "value": sum(x > 0 for x in block_gains), "block_gains": block_gains, "passes": sum(x > 0 for x in block_gains) >= 6},
    ]
    for control in ("space_recent32", "space_recent8"):
        difference = full["found"] - all_means[control]["found"]
        criteria.append({"name": f"no_failure_regression_vs_{control}", "value": difference, "passes": difference >= 0})
    for control in ("space_recent8", "space_initial"):
        difference = all_means[control]["capped_time"] - full["capped_time"]
        criteria.append({"name": f"strict_time_gain_vs_{control}", "value": difference, "passes": difference > 0})
    criteria.extend([{"name": "complete_cohort", "value": len(rows), "passes": True},
                     {"name": "structural_qualification", "value": structural, "passes": structural}])
    return {"scope": "classical odor-memory opportunity screen; no trained architecture or official evaluator replication",
            "episodes": len(rows), "initial_hit_weights": weights, "means": all_means, "strata": strata, "blocks": blocks,
            "criteria": criteria, "learned_pilot_opportunity": all(c["passes"] for c in criteria)}


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.clock = None
        self.start = None
        self.receipt = {"status": "started", "native_steps_attempted": 0, "native_steps_returned": 0,
                        "completed_episodes": 0, "training_updates": 0, "external_model_calls": 0}
        self.launch = None

    def check(self):
        if self.launch is not None:
            require(self.clock.now_ns() < self.launch["deadline_ns"], "native wall cap expired")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = peak
        require(peak <= 4 * 1024**3, "RSS cap exceeded")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= 256 * 1024**2,
                "output cap exceeded")
        require(self.receipt["native_steps_attempted"] <= 340000, "native step cap exceeded")

    def step(self, env, action, hit=None):
        self.check()
        require(self.receipt["native_steps_attempted"] < 340000, "native step allocation exhausted")
        if self.receipt.get("phase") == "qualification":
            require(self.receipt["native_steps_attempted"] < 1000, "qualification step allocation exhausted")
        self.receipt["native_steps_attempted"] += 1
        result = env.step(action, hit=hit, quiet=True)
        self.receipt["native_steps_returned"] += 1
        return result

    def authenticate(self):
        for name, pin in self.plan["sources"].items():
            require(sha(ROOT / name) == pin, f"source changed: {name}")
        require(subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
                == self.plan["upstream_commit"], "upstream commit changed")
        require(not subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "HEAD", "--name-only"], text=True).strip(),
                "tracked upstream modified")
        for name, pin in self.plan["upstream_sources"].items():
            require(sha(UPSTREAM / name) == pin, f"upstream source changed: {name}")
        for name, version in self.plan["runtime_versions"].items():
            require(importlib.metadata.version(name) == version, f"runtime changed: {name}")
        require(sys.version.split()[0] == self.plan["python_version"], "Python version changed")
        extras = subprocess.check_output(["git", "-C", str(UPSTREAM), "ls-files", "--others", "--exclude-standard"], text=True)
        require(not any(p.endswith(".py") for p in extras.splitlines()), "untracked upstream Python source")

    def bind(self):
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "supervisor missing")
            time.sleep(.01)
        self.launch = json.loads(self.args.supervision.read_text())
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                and self.launch["cap_seconds"] == 1800 and self.launch["clock_backend"] == self.clock.backend
                and Path(self.launch["cwd"]).resolve() == ROOT and Path.cwd().resolve() == ROOT
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 1800 * 10**9,
                "supervisor binding mismatch")
        require(sha(self.args.plan) == self.args.plan_sha256, "external plan pin changed")
        self.plan = json.loads(self.args.plan.read_text())
        self.receipt.update(plan_sha256=sha(self.args.plan), supervision_sha256=sha(self.args.supervision),
                            sources=self.plan["sources"], upstream_sources=self.plan["upstream_sources"])
        self.authenticate()
        self.check()

    def qualify(self, SourceTracking, HeuristicPolicy, seeded_environment, observation, OdorMemory):
        import numpy as np
        from scipy.special import k0
        from scipy.stats import poisson

        checks = []

        def record(name, passed, **details):
            checks.append({"name": name, "passes": bool(passed), **details})
            write(self.out / f"qualification-{len(checks):02}.json", checks[-1])
            require(passed, f"qualification failed: {name}")

        template = SourceTracking(**CONFIG, draw_source=False, initial_hit=1)
        record("official_autoset_geometry", template.N == 19 and template.Nhits == 3)
        # Independent analytic likelihood formula for the frozen 2D configuration.
        cells = np.indices((39, 39)) - 19
        distance = np.sqrt(np.sum(cells**2, axis=0))
        mu = k0(np.where(distance == 0, 1, distance)) / np.log(2)
        analytic = np.stack([poisson.pmf(0, mu), poisson.pmf(1, mu), poisson.sf(1, mu)])
        analytic[:, 19, 19] = 0
        error = float(np.max(np.abs(analytic - template.p_Poisson)))
        record("all_likelihoods_include_tail_and_zero_origin", error <= 1e-12, maximum_absolute_difference=error)
        radius = np.arange(1, 1000)
        shell = np.pi * ((radius + .5)**2 - (radius - .5)**2)
        mus = k0(radius) / np.log(2)
        masses = np.array([0, np.sum(poisson.pmf(1, mus) * shell), np.sum(poisson.sf(1, mus) * shell)])
        weights = masses / masses.sum()
        # Check adapter's actual initial-hit distribution against the independent formula.
        random_initial = seeded_environment(SourceTracking, 410001, CONFIG)
        initial_draw = next(d for d in random_initial.draw_log if d["channel"] == "initial")
        actual_weights = np.asarray(initial_draw["probabilities"])
        error = float(np.max(np.abs(weights - actual_weights)))
        record("initial_hit_mixture", error <= 1e-12 and weights[0] == 0 and min(weights[1:]) > 0,
               weights=actual_weights.tolist(), maximum_absolute_difference=error)
        weights = actual_weights
        replay_rows = []
        maximum_filter_error = 0.0

        def forced_environment(source, initial_hit):
            class Forced(SourceTracking):
                def _draw_a_source(self):
                    self.source = source.copy()
            return Forced(**CONFIG, draw_source=True, initial_hit=initial_hit)

        for h in (1, 2):
            first = seeded_environment(SourceTracking, 410010 + h, CONFIG, initial_hit=h)
            second = seeded_environment(SourceTracking, 410010 + h, CONFIG, initial_hit=h)
            original_state = np.random.get_state()
            forced_source = first.source.copy()

            reference = forced_environment(forced_source, h)
            packet = observation(first, 0)
            models = [SourceTracking(**CONFIG, draw_source=False, initial_hit=h) for _ in (0, 1)]
            actors = [OdorMemory(m, HeuristicPolicy, packet, None, i) for i, m in enumerate(models)]
            record(f"initial_prior_and_source_hit{h}", np.array_equal(first.source, second.source)
                   and np.array_equal(first.p_source, reference.p_source) and first.p_source[tuple(first.source)] > 0
                   and packet.hit == h and set(packet._fields) == {"position", "hit", "done", "step", "valid_actions"})
            # Follow a fixed boundary-reaching path, then move directly to the sampled source.
            actions = [0] * 11 + [2] * 11
            for t in range(1, 101):
                if actions:
                    action = actions.pop(0)
                else:
                    differences = first.source - first.agent
                    axis = int(np.flatnonzero(differences)[0])
                    action = 2 * axis + int(differences[axis] > 0)
                for actor in actors:
                    before = actor.model.p_source.copy()
                    a, _ = actor.choose()
                    require(a == int(actor.policy.choose_action()), "upstream policy tie routing changed")
                    require(np.array_equal(before, actor.model.p_source), "hypothetical planning mutated real belief")
                got = self.step(first, action)
                clone = self.step(second, action)
                replay = self.step(reference, action, hit=int(got[0]))
                require(got == clone == replay and np.array_equal(first.p_source, reference.p_source)
                        and np.array_equal(first.p_source, second.p_source), "seeded/native forced-hit replay differs")
                packet = observation(first, t)
                for actor in actors:
                    actor.observe(packet)
                    delta = float(np.max(np.abs(actor.model.p_source - first.p_source)))
                    maximum_filter_error = max(maximum_filter_error, delta)
                    require(delta <= 1e-10, "full public belief differs from native filter")
                replay_rows.append({"initial_hit": h, "step": t, "action": action, "hit": packet.hit,
                                    "position": packet.position, "done": packet.done, "exact_replay": True})
                if got[2]:
                    break
            record(f"seeded_and_forced_hit_replay_hit{h}", bool(first.obs["done"]), steps=t,
                   source=first.source.tolist(), draw_log=first.draw_log)
            new_state = np.random.get_state()
            record(f"global_numpy_rng_unchanged_hit{h}", original_state[0] == new_state[0]
                   and np.array_equal(original_state[1], new_state[1]) and original_state[2:] == new_state[2:])
        record("full_public_filter_and_policy_parity", maximum_filter_error <= 1e-10,
               maximum_absolute_difference=maximum_filter_error, transitions_checked=len(replay_rows))
        # Deliberately injected source/hits are semantics fixtures, not performance samples.
        forced_source = np.array([18, 18])
        fixture = forced_environment(forced_source, 1)
        start = tuple(fixture.agent)
        got = self.step(fixture, 0, hit=2)
        record("saturated_hit_public_update", got == (2, 0, False) and fixture.obs["hit"] == 2
               and np.isclose(fixture.p_source.sum(), 1, atol=1e-10, rtol=0))
        for _ in range(12):
            self.step(fixture, 0, hit=0)
        edge = tuple(fixture.agent)
        self.step(fixture, 0, hit=0)
        record("boundary_action_stays_put", edge == tuple(fixture.agent) and edge[0] == 0
               and not fixture._move(0, fixture.agent)[1], initial_position=start)
        while not fixture.obs["done"]:
            axis = int(np.flatnonzero(fixture.source - fixture.agent)[0])
            action = 2 * axis + int(fixture.source[axis] > fixture.agent[axis])
            self.step(fixture, action, hit=0)
        record("source_found_terminal", fixture.obs == {"hit": -2, "done": True}
               and fixture.p_source[18, 18] == 1 and fixture.p_source.sum() == 1)
        record("qualification_native_step_budget", self.receipt["native_steps_attempted"] <= 1000,
               steps=self.receipt["native_steps_attempted"])
        write(self.out / "qualification.json", {"status": "completed", "checks": checks,
              "replay": replay_rows, "native_steps": self.receipt["native_steps_returned"],
              "scope": "simulator and classical-policy subset; injected semantics fixtures are not efficacy"})
        np.savez_compressed(self.out / "public-kernel.npz", likelihood=template.p_Poisson,
                            initial_hit_weights=weights)
        return {1: float(weights[1]), 2: float(weights[2])}

    def cohort(self, SourceTracking, HeuristicPolicy, seeded_environment, observation, OdorMemory, weights):
        import numpy as np

        rows = []
        max_filter_error = 0.0
        with (self.out / "transitions.jsonl").open("x") as transitions, (self.out / "episodes.jsonl").open("x") as episodes:
            def emit(stream, row):
                stream.write(json.dumps(row, allow_nan=False) + "\n")

            for case in range(64):
                self.check()
                seed, hit, block = 510001 + case, 1 + (case % 8) // 4, case // 8
                shift = case % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    self.receipt["active_case"] = {"seed": seed, "arm": arm}
                    env = seeded_environment(SourceTracking, seed, CONFIG, initial_hit=hit)
                    tick = time.perf_counter()
                    packet = observation(env, 0)
                    model = SourceTracking(**CONFIG, draw_source=False, initial_hit=hit)
                    window = {"full": None, "recent32": 32, "recent8": 8, "initial": 0}[arm.split("_")[1]]
                    actor = OdorMemory(model, HeuristicPolicy, packet, window, 1 if arm.startswith("space") else 0)
                    init_time = time.perf_counter() - tick
                    plan_time = filter_time = env_time = 0.0
                    stuck = 0
                    reset = {"kind": "reset", "seed": seed, "arm": arm, "block": block, "initial_hit": hit,
                             "source_evaluation_only": env.source.tolist(), "public": packet._asdict()}
                    emit(transitions, reset)
                    for t in range(1, HORIZON + 1):
                        self.check()
                        tick = time.perf_counter()
                        action, scores = actor.choose()
                        planning = time.perf_counter() - tick
                        require(action in packet.valid_actions, "policy chose forbidden action")
                        require(np.isfinite(scores[list(packet.valid_actions)]).all(), "nonfinite valid action score")
                        tick = time.perf_counter()
                        _, _, done = self.step(env, action)
                        environment = time.perf_counter() - tick
                        tick = time.perf_counter()
                        packet = observation(env, t)
                        actor.observe(packet)
                        filtering = time.perf_counter() - tick
                        if window is None:
                            error = float(np.max(np.abs(actor.model.p_source - env.p_source)))
                            max_filter_error = max(max_filter_error, error)
                            require(error <= 1e-10, "cohort public full filter differs from native")
                        else:
                            error = None
                        stuck += int(env.agent_stuck)
                        plan_time += planning
                        filter_time += filtering
                        env_time += environment
                        emit(transitions, {"kind": "step", "seed": seed, "arm": arm, "step": t,
                             "action": action, "scores": [float(x) if np.isfinite(x) else None for x in scores],
                             "public": packet._asdict(), "full_filter_max_error": error,
                             "posterior_sha256": hashlib.sha256(actor.model.p_source.tobytes()).hexdigest(),
                             "posterior_mass": float(actor.model.p_source.sum()), "stuck": bool(env.agent_stuck),
                             "planning_seconds": planning, "filter_seconds": filtering, "environment_seconds": environment})
                        if done:
                            break
                    row = {"seed": seed, "arm": arm, "block": block, "initial_hit": hit,
                           "steps": t, "capped_time": t, "found": bool(done), "stuck_steps": stuck,
                           "initialization_seconds": init_time, "planning_seconds": plan_time,
                           "filter_seconds": filter_time, "controller_seconds": init_time + plan_time + filter_time,
                           "environment_seconds": env_time, "source_evaluation_only": env.source.tolist(),
                           "draw_log": env.draw_log}
                    rows.append(row)
                    emit(episodes, row)
                    episodes.flush()
                    transitions.flush()
                    self.receipt["completed_episodes"] = len(rows)
                    print(json.dumps({"episodes": len(rows), "native_steps": self.receipt["native_steps_returned"]}), flush=True)
        summary = summarize(rows, weights, qualification=json.loads((self.out / "qualification.json").read_text()))
        summary["full_filter_max_error"] = max_filter_error
        summary["structural_qualification_passed"] = True
        write(self.out / "summary.json", summary)
        self.receipt["learned_pilot_opportunity"] = summary["learned_pilot_opportunity"]

    def execute(self):
        primary_error = None
        try:
            self.clock = SuspendClock()
            self.start = self.clock.now_ns()
            self.bind()
            sys.path.insert(0, str(UPSTREAM))
            from isotropic.classes.heuristicpolicy import HeuristicPolicy
            from isotropic.classes.sourcetracking import SourceTracking

            from openjev.research.otto_memory import OdorMemory
            from openjev.research.otto_public import observation, seeded_environment
            for name in ("sourcetracking", "heuristicpolicy", "policy"):
                require(Path(sys.modules[f"isotropic.classes.{name}"].__file__).resolve()
                        == UPSTREAM / f"isotropic/classes/{name}.py", "unexpected upstream import path")
            write(self.out / "imports.json", {"python": sys.version, "executable": sys.executable,
                  "versions": {name: importlib.metadata.version(name) for name in self.plan["runtime_versions"]},
                  "upstream": {name: str(sys.modules[name].__file__) for name in
                               ("isotropic.classes.sourcetracking", "isotropic.classes.heuristicpolicy", "isotropic.classes.policy")}})
            self.receipt["phase"] = "qualification"
            weights = self.qualify(SourceTracking, HeuristicPolicy, seeded_environment, observation, OdorMemory)
            self.receipt["qualification_steps"] = self.receipt["native_steps_returned"]
            self.receipt["phase"] = "cohort"
            self.cohort(SourceTracking, HeuristicPolicy, seeded_environment, observation, OdorMemory, weights)
            self.authenticate()
            self.check()
            self.receipt["status"] = "completed"
        except BaseException as error:
            primary_error = error
            self.receipt.update(status="failed", error=repr(error))
            raise
        finally:
            finalization_errors = []
            try:
                self.receipt["native_elapsed_seconds"] = (self.clock.now_ns() - self.start) / 1e9
            except BaseException as error:  # noqa: BLE001 - preserve primary failure during cleanup
                self.receipt["native_elapsed_seconds"] = None
                finalization_errors.append(f"clock finalization: {error!r}")
            self.receipt["timing_scope"] = "worker before receipt publication; authoritative supervisor includes process lifetime"
            try:
                self.receipt["files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                         for p in self.out.iterdir() if p.is_file()}
            except BaseException as error:  # noqa: BLE001 - preserve primary failure during cleanup
                finalization_errors.append(f"artifact hashing: {error!r}")
            if finalization_errors:
                self.receipt.update(status="failed", finalization_errors=finalization_errors)
            try:
                write(self.out / "receipt.json", self.receipt)
            except BaseException as error:  # noqa: BLE001 - preserve primary failure during cleanup
                finalization_errors.append(f"receipt publication: {error!r}")
                print(json.dumps({"status": "failed", "primary_error": repr(primary_error),
                                  "finalization_errors": finalization_errors}), file=sys.stderr, flush=True)
            if finalization_errors and primary_error is None:
                raise RuntimeError("; ".join(finalization_errors))
        try:
            self.check()
        except BaseException as error:
            self.receipt.update(status="failed", error=f"late publication check: {error!r}")
            try:
                write(self.out / "late-failure.json", self.receipt)
            except OSError as publication_error:
                print(f"late-failure publication: {publication_error!r}", file=sys.stderr, flush=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    Run(parser.parse_args()).execute()
