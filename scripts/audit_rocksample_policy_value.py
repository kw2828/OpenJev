"""Independent saved-transition accounting for the classical RockSample screen.

No simulator, belief, planner or producer scoring imports. Recorded posterior
diagnostics and forecast values are authenticated witnesses, not recomputed
Bayesian inference or proof of planner optimality. Raw return and all ten gates
are independently reconstructed from complete public transition/decision joins.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import resource
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

ARMS = ("exit", "full", "recent128", "latest", "quality", "privileged")
MAPS = tuple(range(12001, 12009))
RESETS = tuple(range(22001, 22005))
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
RUNTIME_PIN = "b233dbd09714ae574626cdd2f58b3efde80c0127759af8f399147845ece4bcb2"
FILES = {"transitions.jsonl", "decisions.jsonl", "episodes.jsonl", "summary.json"}
METRICS = ("raw_return", "discounted_return_099", "steps", "checks", "samples", "good_samples",
           "bad_samples", "empty_samples", "controller_seconds", "filter_seconds", "planning_seconds",
           "environment_seconds", "plans", "exited", "truncated")
WORK = {"non_sensing_calls", "expected_reward_grid_calls", "tour_evaluations",
        "positive_target_visits_considered", "feasible_tour_prefixes", "vantages_considered",
        "vantages_feasible", "entropy_probability_calls", "bundle_candidates", "bundles_evaluated",
        "bundles_infeasible", "check_probability_calls", "condition_check_calls",
        "outcome_branches_considered", "zero_probability_branches_skipped", "outcome_leaves",
        "maximum_leaves_per_bundle", "maximum_branch_mass_error"}
SCOPE = ("Independent complete public trace/macro/depletion joins, reward and discount arithmetic, "
         "recorded work and ESS aggregation, eight-map means and ten gates. Posterior values, "
         "forecast optimality, actual random generation, simulator execution, hidden-reference "
         "contents and timing/RSS truth remain inherited from authenticated source/runtime/process "
         "witnesses. No environment, model or training calls.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"integer: {name}")
    return value


def finite(value, name, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"finite scalar: {name}")
    require(minimum is None or value >= minimum, f"lower bound: {name}")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Comparison:
    def __init__(self):
        self.scalars = 0
        self.maximum_error = 0.0

    def same(self, actual, expected, label):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), f"keys: {label}")
            for key, value in expected.items():
                self.same(actual[key], value, f"{label}.{key}")
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"length: {label}")
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.same(left, right, f"{label}[{index}]")
        else:
            self.scalars += 1
            if type(expected) is float:
                finite(actual, label)
                difference = abs(actual - expected)
                self.maximum_error = max(self.maximum_error, difference)
                require(difference <= 1e-12, f"numeric disagreement: {label}")
            else:
                require(type(actual) is type(expected) and actual == expected, f"exact disagreement: {label}")


def authenticate(args, budget):
    run, plan_path, terminal_path = args.run.resolve(), args.plan.resolve(), args.terminal.resolve()
    require(sha(plan_path) == args.plan_sha256, "external plan pin")
    require(sha(run / "receipt.json") == args.receipt_sha256, "external run receipt pin")
    require(sha(terminal_path) == args.terminal_sha256, "external parent terminal pin")
    plan, receipt, terminal = read(plan_path), read(run / "receipt.json"), read(terminal_path)
    require(receipt["status"] == "completed" and terminal["status"] == "completed", "completed run required")
    require({p.name for p in run.iterdir()} == FILES | {"receipt.json"}, "exact five-file run closure")
    require(set(receipt["files"]) == FILES, "exact four payloads")
    for name, witness in receipt["files"].items():
        path = run / name
        require(not path.is_symlink() and path.stat().st_size == witness["bytes"]
                and sha(path) == witness["sha256"], f"payload authentication: {name}")
        budget()
    require(sum(p.stat().st_size for p in run.iterdir()) <= 512 * 1024**2, "saved output cap")
    require(receipt["plan_sha256"] == args.plan_sha256 and receipt["sources"] == plan["sources"], "plan/source joins")
    for name, pin in plan["sources"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "relative source path")
        require(sha(ROOT / name) == pin, f"source authentication: {name}")
    require(plan["sources"]["src/openjev/research/suspend_clock.py"] == CLOCK_PIN, "clock source pin")
    require(plan["runtime_qualification_sha256"] == RUNTIME_PIN, "fixed runtime qualification")
    qualified = ROOT / "output/rocksample-runtime-v1/qualification-01"
    require(sha(qualified / "completed.json") == RUNTIME_PIN, "qualification completion")
    qualification = read(qualified / "completed.json")
    for name, witness in qualification["files"].items():
        path = qualified / name
        require(path.stat().st_size == witness["bytes"] and sha(path) == witness["sha256"], f"runtime payload: {name}")
    runtime_terminal = qualified.parent / "qualification-process-01.terminal.json"
    require(sha(runtime_terminal) == plan["runtime_terminal_sha256"], "qualification terminal pin")
    require(read(runtime_terminal)["status"] == "completed", "qualification terminal success")
    imports = read(qualified / "imports.json")
    require(sha(imports["gymnax_environment"]["path"]) == imports["gymnax_environment"]["sha256"], "Gymnax source")
    for name, pin in imports["upstream_imported_source_sha256"].items():
        require(sha(ROOT / "tmp/pobax-source-review-01" / name) == pin, f"upstream source: {name}")
    require(plan["cohort"] == {"map_seeds": list(MAPS), "reset_seeds": list(RESETS),
                              "arms": list(ARMS), "episodes": 192, "transition_seed_base": 430000,
                              "transition_seed_rule": "base + map_index*4 + reset_index; shared across arms for each case",
                              "arm_order": "rotate by case index modulo six", "particle_bank_seed": 530001,
                              "particles": 256, "episode_horizon": 1000, "check_cap": 64}, "fixed cohort declaration")
    require(plan["limits"] == {"wall_seconds": 1800, "rss_bytes": 8 * 1024**3,
                              "output_bytes": 512 * 1024**2, "primitive_transitions": 192000}, "fixed limits")
    for name in ("qualification", "timing"):
        require(sha(ROOT / plan["engineering"][name]) == plan["engineering"][name + "_sha256"], f"engineering pin: {name}")
    launch_path = terminal_path.with_name(terminal_path.name.replace(".terminal.json", ".launch.json"))
    require(launch_path != terminal_path and sha(launch_path) == receipt["supervision_sha256"], "launch pin")
    launch = read(launch_path)
    command = list(launch["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    expected = [str(ROOT / "tmp/pobax-runtime-02/bin/python"), str(ROOT / "scripts/study_rocksample_policy_value.py"),
                "--output", str(run), "--supervision", str(launch_path), "--plan", str(plan_path),
                "--plan-sha256", args.plan_sha256]
    require(command == expected and terminal["command"] == launch["command"], "exact supervised command")
    for key in ("version", "cwd", "clock_backend", "started_ns", "deadline_ns", "cap_seconds", "pid", "pgid",
                "parent_pid", "watchdog_sha256", "clock_source_sha256"):
        require(terminal[key] == launch[key], f"supervisor join: {key}")
    require(Path(launch["cwd"]).resolve() == ROOT and launch["version"] == "dialogue-observation-supervision-v2", "supervisor identity")
    require(launch["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME"), "native clock")
    require(launch["clock_source_sha256"] == CLOCK_PIN, "supervisor clock source")
    require(launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"], "watchdog source")
    start = integer(launch["started_ns"], "parent start")
    finish = integer(terminal["finished_ns"], "parent finish")
    elapsed = integer(terminal["elapsed_ns"], "parent elapsed")
    require(launch["cap_seconds"] == 1800 and launch["deadline_ns"] == start + 1800 * 10**9, "parent deadline")
    require(start <= finish < launch["deadline_ns"] and elapsed == finish - start, "strict parent elapsed")
    require(terminal["wall_seconds"] == elapsed / 1e9 and terminal["timing_available"] is True, "elapsed units")
    require(terminal["returncode"] == 0 and terminal["group_absent"] is True and terminal["timed_out"] is False
            and terminal["error"] is None and terminal["clock_error"] is None, "successful actual exit")
    require(terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["errors"] == [], "cleanup witnesses")
    require(launch["pid"] == launch["pgid"] and launch["pid"] != launch["parent_pid"], "isolated child group")
    for record in (launch, terminal):
        finite(record["started_unix"], "civil provenance", 0)
    finite(terminal["finished_unix"], "civil provenance", 0)
    require(0 <= finite(receipt["native_elapsed_seconds"], "worker elapsed") <= terminal["wall_seconds"], "worker within parent")
    require(integer(receipt["peak_rss_bytes"], "worker RSS") <= 8 * 1024**3, "worker memory cap")
    require(receipt["training_updates"] == receipt["external_model_calls"] == 0, "no training/model calls")
    budget()
    return plan, receipt, terminal


def identity(row):
    return row["map_seed"], row["reset_seed"], row["arm"]


def groups(path):
    with path.open() as stream:
        rows = (json.loads(line) for line in stream)
        for key, members in itertools.groupby(rows, key=identity):
            yield key, list(members)


def position(observation, label):
    require(isinstance(observation, list) and len(observation) == 33, f"raw33: {label}")
    for value in observation:
        finite(value, label)
    result = []
    for channel in (observation[:11], observation[11:22]):
        require(all(value in (0, 1) for value in channel) and sum(channel) == 1, f"one hot: {label}")
        result.append(channel.index(1))
    require(all(value in (-1, 0, 1) for value in observation[22:]), f"signed readings: {label}")
    return tuple(result)


def moves(start, target):
    dy, dx = target[0] - start[0], target[1] - start[1]
    return [2 if dy > 0 else 0] * abs(dy) + [1 if dx > 0 else 3] * abs(dx)


def check_plan(plan, coordinate, ledger, remaining, check_budget, arm, compare):
    actions = plan["actions"]
    require(actions and len(actions) <= remaining and all(type(a) is int and 0 <= a < 16 for a in actions), "legal full macro")
    require(sum(a >= 5 for a in actions) <= check_budget, "global check budget")
    kind = plan["kind"]
    require(kind in ("exit", "sense", "exploit"), "macro kind")
    if kind == "exit":
        require(actions == [1] * min(10 - coordinate[1], remaining), "direct east exit")
    if arm == "exit":
        require(set(plan) == {"kind", "actions", "work"} and kind == "exit" and plan["work"] == {}, "exit controller")
        return
    compare.same(plan["committed_steps"], len(actions), "macro duration")
    work = plan["work"]
    require(set(work) == WORK, "complete planner work fields")
    for name, value in work.items():
        if name == "maximum_branch_mass_error":
            require(0 <= finite(value, name) <= 1e-12, "branch mass witness")
        else:
            integer(value, name)
    require(work["bundles_evaluated"] <= 25 and work["outcome_leaves"] <= 210
            and work["maximum_leaves_per_bundle"] <= 16, "fixed search expansion bounds")
    require(work["non_sensing_calls"] == work["expected_reward_grid_calls"] == 1 + work["outcome_leaves"], "continuation counts")
    require(work["tour_evaluations"] == 4 * work["non_sensing_calls"], "four tours per continuation")
    require(work["bundle_candidates"] == work["bundles_evaluated"] + work["bundles_infeasible"], "bundle partition")
    require(work["condition_check_calls"] == work["outcome_branches_considered"] - work["zero_probability_branches_skipped"], "condition count")
    require(2 * (work["check_probability_calls"] - work["entropy_probability_calls"]) == work["outcome_branches_considered"], "binary branching count")
    require(work["entropy_probability_calls"] == 11 * work["vantages_feasible"], "all-rock entropy count")
    require(work["vantages_considered"] == len({coordinate, (0, 0), (0, 9), (10, 0), (10, 9)}), "public vantages")
    require(plan["check_margin"] == 0.05, "fixed sensing margin")
    baseline = finite(plan["non_sensing_return"], "base forecast")
    value = finite(plan["expected_return"], "selected forecast")
    if plan["best_sensing_return"] is not None:
        compare.same(plan["best_sensing_gain"], float(plan["best_sensing_return"] - baseline), "best sensing gain")
    else:
        require(plan["best_sensing_gain"] is None and work["bundles_evaluated"] == 0, "absent sensing")
    if kind == "sense":
        target = tuple(plan["sensing_position"])
        require(target in {coordinate, (0, 0), (0, 9), (10, 0), (10, 9)}, "sensing vantage")
        rocks = plan["check_rocks"]
        require(len(rocks) in (1, 2, 4) and all(type(r) is int and 0 <= r < 11 for r in rocks), "sensing bundle")
        require(len(set(rocks)) in (1, len(rocks)), "fixed mixed/repeated family")
        require(actions == moves(coordinate, target) + [5 + r for r in rocks], "committed sensing movement/readings")
        require(len(actions) + 10 - target[1] <= remaining, "post-sensing exit reserved")
        require(value > baseline + 0.05 and value == plan["best_sensing_return"], "sensing strictly earns margin")
        compare.same(plan["sensing_gain"], float(value - baseline), "chosen sensing gain")
        compare.same(plan["checks_committed"], len(rocks), "chosen check count")
    else:
        require(value == baseline and plan["sensing_gain"] == 0 and plan["checks_committed"] == 0
                and plan["check_rocks"] == [], "non-sensing identity")
        require(plan["best_sensing_return"] is None or plan["best_sensing_return"] <= baseline + 0.05, "no skipped winning sensing")
        if kind == "exploit":
            target = tuple(plan["first_sample_target"])
            require(len(target) == 2 and all(type(v) is int for v in target)
                    and 0 <= target[0] < 11 and 0 <= target[1] < 10 and target not in ledger, "new public sample target")
            require(actions == moves(coordinate, target) + [4], "first-sample-only commitment")
            require(len(actions) + 10 - target[1] <= plan["forecast_tour_steps"] <= remaining, "exploitation exit/horizon")


def episode(key, traces, decisions, reported, compare):
    require(identity(reported) == key and traces and traces[0]["step"] == -1 and traces[0]["event"] == "reset", "episode reset identity")
    coordinate = position(traces[0]["observation"], "reset")
    require(coordinate[1] < 10 and not any(traces[0]["observation"][22:]), "fresh public reset")
    ledger, rewards, work = set(), [], {}
    checks = good = bad = empty = adequate = ess_count = 0
    minimum_ess = maximum_weight = None
    kinds = {"sense": 0, "exploit": 0, "exit": 0}
    planning = []
    index, ended, truncated = 0, False, False
    arm = key[2]
    for plan_index, decision in enumerate(decisions):
        require(not ended and decision["step"] == index, "decision at next incomplete step")
        compare.same(decision["position"], list(coordinate), "decision public position")
        compare.same(decision["remaining_steps"], 1000 - index, "remaining horizon")
        compare.same(decision["checks_remaining"], 64 - checks, "remaining checks")
        compare.same(decision["sampled_cells"], len(ledger), "persistent depletion ledger")
        diag = decision["belief_diagnostics"]
        if arm in ("full", "recent128", "latest", "privileged"):
            particles = 1 if arm == "privileged" else 256
            ess = finite(diag["ess"], "recorded ESS", 1 - 1e-12)
            weight = finite(diag["max_weight"], "recorded maximum weight", 0)
            require(ess <= particles + 1e-12 and 1 / particles - 1e-12 <= weight <= 1
                    and diag["particles"] == particles and 1 <= integer(diag["alive_particles"], "live maps") <= particles, "particle witness bounds")
            ess_count += 1
            adequate += ess >= 8
            minimum_ess = ess if minimum_ess is None else min(minimum_ess, ess)
            maximum_weight = weight if maximum_weight is None else max(maximum_weight, weight)
        else:
            require(diag["ess"] is None, "no particle ESS for this arm")
            if arm == "quality":
                compare.same(diag["sampled_cells"], len(ledger), "quality depletion ledger")
        planning.append(finite(decision["plan_seconds"], "planning time", 0))
        plan = decision["plan"]
        check_plan(plan, coordinate, ledger, 1000 - index, 64 - checks, arm, compare)
        kinds[plan["kind"]] += 1
        for name, value in plan["work"].items():
            work[name] = max(work.get(name, 0), value) if name.startswith("maximum_") else work.get(name, 0) + value
        for ai, action in enumerate(plan["actions"]):
            require(index + 1 < len(traces), "every committed action returned")
            record = traces[index + 1]
            require(record["step"] == index and record["action"] == action and record["plan_index"] == plan_index, "primitive/macro join")
            new = position(record["observation"], "returned observation")
            expected = coordinate
            if action < 4:
                dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
                expected = (min(10, max(0, coordinate[0] + dy)), min(10, max(0, coordinate[1] + dx)))
            natural = expected[1] == 10
            horizon = index + 1 >= 1000
            ended, truncated = natural or horizon, horizon and not natural
            compare.same(record["done"], ended, "natural/timeout boundary")
            compare.same(record["truncated"], truncated, "natural termination precedence")
            compare.same(record["time_limit_reached"], horizon, "native time limit")
            reward = finite(record["reward"], "native reward")
            require(reward in (-10, 0, 10), "native reward support")
            if action == 4:
                require(coordinate not in ledger, "no repeat sampling")
                ledger.add(coordinate)
                good += reward == 10
                bad += reward == -10
                empty += reward == 0
            else:
                require(reward == (10 if natural else 0), "movement/check reward")
            readings = record["observation"][22:]
            if ended:
                require(ai == len(plan["actions"]) - 1 and new[1] < 10 and not any(readings), "full macro then keyed reset")
            else:
                require(new == expected, "public motion semantics")
                active = [i for i, value in enumerate(readings) if value != 0]
                require(active == ([action - 5] if action >= 5 else []), "ephemeral selected check only")
            checks += action >= 5
            rewards.append(float(reward))
            coordinate = new
            index += 1
    require(ended and 0 < index <= 1000 and len(traces) == index + 1 and checks <= 64, "complete episode coverage")
    require(good + bad <= 11, "at most eleven nonempty first samples")
    filter_time = finite(reported["filter_seconds"], "recorded filtering time", 0)
    env_time = finite(reported["environment_seconds"], "recorded environment time", 0)
    expected = {"map_seed": key[0], "reset_seed": key[1], "arm": arm,
                "raw_return": math.fsum(rewards), "discounted_return_099": math.fsum((.99 ** t) * r for t, r in enumerate(rewards)),
                "steps": index, "checks": checks, "samples": len(ledger), "good_samples": good,
                "bad_samples": bad, "empty_samples": empty, "exited": not truncated, "truncated": truncated,
                "plans": len(decisions), "plan_kinds": kinds, "work": work,
                "filter_seconds": filter_time, "planning_seconds": math.fsum(planning),
                "controller_seconds": filter_time + math.fsum(planning), "environment_seconds": env_time,
                "ess_adequate_plans": adequate, "ess_evaluated_plans": ess_count,
                "minimum_ess": minimum_ess, "maximum_map_weight": maximum_weight,
                "information": "true map and qualities" if arm == "privileged" else "public history only"}
    compare.same(reported, expected, f"episode{key}")
    return expected


def mean(values):
    return math.fsum(values) / len(values)


def aggregate(rows):
    maps = []
    for map_seed in MAPS:
        scores = {arm: {metric: mean([row[metric] for row in rows if row["map_seed"] == map_seed and row["arm"] == arm])
                        for metric in METRICS} for arm in ARMS}
        maps.append({"map_seed": map_seed, "scores": scores})
    averages = {arm: {metric: mean([panel["scores"][arm][metric] for panel in maps]) for metric in METRICS} for arm in ARMS}
    decisions = []
    for control in ("recent128", "latest", "quality"):
        differences = [panel["scores"]["full"]["raw_return"] - panel["scores"][control]["raw_return"] for panel in maps]
        gap = averages["full"]["raw_return"] - averages[control]["raw_return"]
        positives = len([value for value in differences if value > 0])
        decisions.extend([{"name": f"raw_gain_vs_{control}", "value": gap, "passes": gap >= 5},
                          {"name": f"positive_maps_vs_{control}", "value": positives, "map_gains": differences, "passes": positives >= 6}])
    for arm, margin in (("full", 5), ("privileged", 20)):
        gap = averages[arm]["raw_return"] - averages["exit"]["raw_return"]
        decisions.append({"name": f"{arm}_gain_vs_exit", "value": gap, "passes": gap >= margin})
    full = [row for row in rows if row["arm"] == "full"]
    adequate = sum(row["ess_adequate_plans"] for row in full)
    count = sum(row["ess_evaluated_plans"] for row in full)
    decisions.extend([{"name": "finite_particle_adequacy", "value": adequate / count if count else 0,
                       "numerator": adequate, "denominator": count, "passes": count > 0 and adequate * 10 >= count * 9},
                      {"name": "complete_cohort", "value": len(rows), "passes": True}])
    return {"scope": "classical policy-value screen, not a trained architecture result", "episodes": len(rows),
            "scores_equal_map_means": averages, "by_map": maps, "criteria": decisions,
            "learned_pilot_admitted": all(item["passes"] for item in decisions)}


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    own_source = sha(__file__)
    comparison = Comparison()
    clock = None
    start = last = None
    try:
        require(sha(ROOT / "src/openjev/research/suspend_clock.py") == CLOCK_PIN, "audit clock source")
        clock = SuspendClock()
        start = clock.now_ns()
        deadline = start + 300 * 10**9

        def budget():
            nonlocal last
            last = clock.now_ns()
            require(last < deadline, "audit native deadline")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            require(rss <= 4 * 1024**3, "audit RSS cap")
            require(sum(p.stat().st_size for p in output.iterdir()) <= 128 * 1024**2, "audit output cap")

        write(output / "started.json", {"source_sha256": own_source, "request": {k: str(v) for k, v in vars(args).items()},
                                       "clock_backend": clock.backend, "started_ns": start, "deadline_ns": deadline,
                                       "scope": SCOPE})
        _plan, receipt, terminal = authenticate(args, budget)
        run = args.run.resolve()
        episodes = groups(run / "episodes.jsonl")
        traces = groups(run / "transitions.jsonl")
        decisions = groups(run / "decisions.jsonl")
        expected_order = []
        for mi, map_seed in enumerate(MAPS):
            for ri, reset_seed in enumerate(RESETS):
                offset = (mi * 4 + ri) % len(ARMS)
                expected_order.extend((map_seed, reset_seed, arm) for arm in ARMS[offset:] + ARMS[:offset])
        rows = []
        for key in expected_order:
            budget()
            eg, tg, dg = next(episodes, None), next(traces, None), next(decisions, None)
            require(eg is not None and tg is not None and dg is not None, "all 192 cases present")
            require(eg[0] == tg[0] == dg[0] == key and len(eg[1]) == 1, "fixed rotated case order and uniqueness")
            rows.append(episode(key, tg[1], dg[1], eg[1][0], comparison))
        require(next(episodes, None) is None and next(traces, None) is None and next(decisions, None) is None, "no extra cases")
        steps = sum(row["steps"] for row in rows)
        require(receipt["steps_attempted"] == receipt["steps_returned"] == steps <= 192000, "root primitive accounting")
        require(receipt["completed_episodes"] == len(rows) == 192, "root episode accounting")
        summary = aggregate(rows)
        comparison.same(read(run / "summary.json"), summary, "complete summary")
        require(receipt["learned_pilot_admitted"] is summary["learned_pilot_admitted"], "receipt decision")
        require(sha(__file__) == own_source, "audit source unchanged")
        for name, witness in receipt["files"].items():
            require(sha(run / name) == witness["sha256"], f"payload unchanged: {name}")
        budget()
        write(output / "summary.json", {"agreement": True, "scope": SCOPE, "episodes": len(rows),
                                       "primitive_transitions": steps, "decisions": sum(row["plans"] for row in rows),
                                       "scalar_comparisons": comparison.scalars, "maximum_absolute_difference": comparison.maximum_error,
                                       "recomputed_summary": summary, "environment_calls": 0, "model_calls": 0})
        budget()
        result = {"status": "completed", "agreement": True, "source_sha256": own_source,
                  "plan_sha256": args.plan_sha256, "producer_receipt_sha256": args.receipt_sha256,
                  "producer_summary_sha256": receipt["files"]["summary.json"]["sha256"],
                  "terminal_sha256": args.terminal_sha256, "parent_wall_seconds": terminal["wall_seconds"],
                  "clock_backend": clock.backend, "elapsed_ns": last - start, "wall_seconds": (last - start) / 1e9,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                  "limits": {"native_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2},
                  "scalar_comparisons": comparison.scalars, "maximum_absolute_difference": comparison.maximum_error,
                  "learned_pilot_admitted": summary["learned_pilot_admitted"], "scope": SCOPE,
                  "timing_scope": "Native start through last pre-receipt check; publication rechecked before return.",
                  "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}}
        write(output / "receipt.json", result)
        budget()
        print(json.dumps({"status": "completed", "agreement": True, "scalar_comparisons": comparison.scalars,
                          "learned_pilot_admitted": summary["learned_pilot_admitted"]}), flush=True)
    except BaseException as exc:
        try:
            if (output / "receipt.json").exists():
                (output / "receipt.json").rename(output / "invalid-receipt.json")
            write(output / "failed.json", {"status": "failed", "agreement": False, "error": repr(exc),
                                           "traceback": traceback.format_exc(), "source_sha256": own_source,
                                           "last_clock_elapsed_ns": None if last is None else last - start,
                                           "timing_available": False, "wall_seconds": None,
                                           "scalar_comparisons_before_failure": comparison.scalars})
        except BaseException as publication_error:  # noqa: BLE001 - preserve the original failure
            exc.add_note(f"Failure evidence publication also failed: {publication_error!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
