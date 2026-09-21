"""Independent saved-only audit of autonomous OTTO spectral-control outcomes.

Uses only the Python standard library and the frozen suspend-inclusive clock.
Does not import a study runner, actor, simulator, or numerical library. Recorded
posterior/parity witnesses and measured cost truth remain source-bound evidence;
this reader independently verifies identities, transitions, cost arithmetic,
weighted outcomes and prospective candidate conditions.
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

CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
ARMS = ("full_bayes", "exact_log", "recent32", "recent32_hard", "dct16_neutral", "dct16_nearest")
CANDIDATES = ("dct16_neutral", "dct16_nearest")
COHORTS = ("base", "shift")
FIRST_SEEDS = {"base": 630001, "shift": 640001}
N, NHITS, HORIZON = 53, 4, 2188
CENTER = (26, 26)
FILES = {"started.json", "imports.json", "qualification.json", "qualification.jsonl",
         "public-kernel-base.npz", "public-kernel-shift.npz", "transitions.jsonl", "episodes.jsonl", "summary.json"}
LIMITS = {"native_seconds": 900, "rss_bytes": 4 * 1024**3, "output_bytes": 512 * 1024**2,
          "native_steps": 2522624, "qualification_steps": 2048}
PINNED = {
    "scripts/study_otto_spectral_memory.py": "9bae9a6095061fc7f6c203ec34bdf19d174e7d13822e348231b98c8884447f88",
    "scripts/audit_otto_large_memory.py": "1ad5a080f58177751dadf0eaab0cd9e2c7b37d7e9d84b50cc32f4046a66155ec",
    "src/openjev/research/otto_spectral_memory.py": "440c7527a6040fb9b41985dc8f48e5f62e18be880242de09f70b46b540579943",
    "src/openjev/research/otto_public.py": "438631a18005493e0cafa0777315e2158cac97cc7d0e71ad9fd6402284615b3d",
    "src/openjev/research/suspend_clock.py": CLOCK_PIN,
    "scripts/supervise_dialogue_observation_v2.py": "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144",
}
TIMES = ("actor_initialization_seconds", "update_seconds", "decode_seconds", "planner_seconds",
         "shared_initialization_allocation_seconds")
METRICS = ("capped_time", "found", "stuck_steps", *TIMES, "controller_seconds",
           "environment_initialization_seconds", "environment_seconds", "episode_seconds", "state_array_bytes")
SCOPE = ("Independent saved-output verification of all 1,152 autonomous episodes: source/process/payload bindings, "
         "paired case identities and recorded random-channel coupling, legal movement and terminal semantics, "
         "recorded action tie rules, recorded cost sums, regime-specific mixture and block means, and frozen "
         "candidate criteria. No posterior or score recomputation, simulator/actor calls, PCG regeneration, "
         "or independent timing/RSS measurement. Qualification arithmetic witnesses remain inherited.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"integer {name}")
    return value


def finite(value, name, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"finite {name}")
    require(minimum is None or value >= minimum, f"minimum {name}")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"SHA256 {name}")
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe_file(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts, "safe manifest path")
    path = root / relative
    require(path.is_file() and all(not component.is_symlink() for component in (path, *path.parents)
                                 if component != root.parent), f"regular non-symlink file {name}")
    require(path.resolve().is_relative_to(root.resolve()), f"contained file {name}")
    return path


class Comparison:
    def __init__(self):
        self.scalars = 0
        self.maximum_error = 0.0

    def close(self, actual, expected, name, tolerance=1e-9):
        finite(actual, name)
        finite(expected, name)
        difference = abs(actual - expected)
        self.scalars += 1
        self.maximum_error = max(self.maximum_error, difference)
        require(difference <= tolerance, f"arithmetic disagreement {name}: {actual!r} != {expected!r}")

    def same(self, actual, expected, name):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), f"keys {name}")
            for key, value in expected.items():
                self.same(actual[key], value, f"{name}.{key}")
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"length {name}")
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.same(left, right, f"{name}[{index}]")
        elif type(expected) is float:
            self.close(actual, expected, name)
        else:
            self.scalars += 1
            require(type(actual) is type(expected) and actual == expected, f"exact {name}")


def grouped(path):
    with Path(path).open() as stream:
        records = (json.loads(line) for line in stream)
        for key, values in itertools.groupby(records, key=lambda row: (row["cohort"], row["seed"], row["arm"])):
            yield key, list(values)


def moved(position, action):
    integer(action, "action")
    require(action < 4, "four action IDs")
    result = list(position)
    axis = action // 2
    result[axis] = max(0, min(N - 1, result[axis] + (-1 if action % 2 == 0 else 1)))
    return tuple(result)


def coordinate(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(x) is int and 0 <= x < N for x in value), "grid coordinate")
    return tuple(value)


def packet(value, step, position, initial_hit=None):
    require(isinstance(value, dict) and set(value) == {"position", "hit", "done", "step", "valid_actions"},
            "public observation whitelist")
    require(coordinate(value["position"]) == position and integer(value["step"], "public step") == step,
            "public time/position join")
    require(type(value["done"]) is bool and type(value["hit"]) is int, "typed public hit/done")
    hit, done = value["hit"], value["done"]
    require(hit == -2 if done else 0 <= hit < NHITS, "terminal sentinel or public odor")
    expected_actions = [] if done else [action for action in range(4) if moved(position, action) != position]
    require(value["valid_actions"] == expected_actions, "public valid-action IDs")
    if initial_hit is not None:
        require(step == 0 and not done and hit == initial_hit and position == CENTER, "initial conditioned observation")
    return hit, done


def selected_action(scores, position):
    require(isinstance(scores, list) and len(scores) == 4, "four recorded action scores")
    valid = [action for action in range(4) if moved(position, action) != position]
    require(all((scores[action] is None) == (action not in valid) for action in range(4)), "action score mask")
    for action in valid:
        finite(scores[action], "action score")
    best = min(scores[action] for action in valid)
    return next(action for action in valid if abs(scores[action] - best) < 1e-10)


def draw_record(row, channel, index, selected, comparison):
    """Validate a recorded categorical draw, without regenerating random numbers."""
    require(set(row) == {"channel", "index", "probabilities", "cdf_mass", "uniform", "selected_index"},
            "recorded draw fields")
    require(row["channel"] == channel and integer(row["index"], "draw index") == index
            and integer(row["selected_index"], "selected index") == selected, "draw identity")
    values = row["probabilities"]
    require(isinstance(values, list) and len(values) == (N * N if channel == "source" else NHITS), "draw support size")
    for probability in values:
        finite(probability, "categorical probability", 0)
    comparison.close(math.fsum(values), 1.0, "categorical mass", 1e-10)
    cumulative, selected_from_cdf = 0.0, None
    uniform = finite(row["uniform"], "recorded uniform", 0)
    require(uniform < 1, "recorded uniform less than one")
    mass = sum(values)
    comparison.close(row["cdf_mass"], mass, "recorded cumulative mass", 1e-10)
    for i, probability in enumerate(values):
        cumulative += probability
        if selected_from_cdf is None and cumulative / mass > uniform:
            selected_from_cdf = i
    require(selected_from_cdf == selected, "recorded inverse CDF choice")
    return uniform


def authenticate_process(args, plan, done, imports):
    terminal_path = args.terminal.resolve()
    require(sha(terminal_path) == args.terminal_sha256, "external supervisor terminal pin")
    terminal = read(terminal_path)
    launch_path = terminal_path.with_name(terminal_path.name.replace(".terminal.json", ".launch.json"))
    require(launch_path != terminal_path and not launch_path.is_symlink(), "supervisor launch path")
    require(sha(launch_path) == done["supervision_sha256"], "worker launch binding")
    launch = read(launch_path)
    for key in ("version", "command", "cwd", "clock_backend", "started_ns", "deadline_ns", "cap_seconds",
                "pid", "pgid", "parent_pid", "watchdog_sha256", "clock_source_sha256"):
        require(launch[key] == terminal[key], f"supervisor launch/terminal {key}")
    command = list(launch["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:2] == [imports["executable"], str(ROOT / "scripts/study_otto_spectral_control.py")]
            and len(command) == 10, "exact worker executable/script and four flags")
    require(dict(zip(command[2::2], command[3::2], strict=True)) == {
        "--plan": str(args.plan.resolve()), "--plan-sha256": args.plan_sha256,
        "--supervision": str(launch_path), "--output": str(args.run.resolve())}, "worker CLI identity")
    require(launch["version"] == "dialogue-observation-supervision-v2" and Path(launch["cwd"]).resolve() == ROOT,
            "supervisor version and working directory")
    require(launch["clock_backend"] in {"mach_continuous_time", "CLOCK_BOOTTIME"}
            and launch["clock_source_sha256"] == CLOCK_PIN
            and launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"],
            "supervisor native clock and source binding")
    require(integer(launch["pid"], "worker PID", 1) == launch["pgid"]
            and launch["pid"] != integer(launch["parent_pid"], "parent PID", 1), "separate worker process group")
    start, finish = integer(launch["started_ns"], "parent start"), integer(terminal["finished_ns"], "parent finish")
    require(launch["cap_seconds"] == 900 and launch["deadline_ns"] == start + 900 * 10**9
            and start <= finish < launch["deadline_ns"] and terminal["elapsed_ns"] == finish - start
            and terminal["wall_seconds"] == (finish - start) / 1e9, "strict parent native deadline and arithmetic")
    require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timing_available"] is True
            and terminal["timed_out"] is False and terminal["group_absent"] is True
            and terminal["error"] is None and terminal["clock_error"] is None, "actual successful supervisor terminal")
    require(terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["errors"] == [], "successful process group cleanup")
    require(0 <= finite(done["native_elapsed_seconds"], "worker elapsed") <= terminal["wall_seconds"],
            "worker elapsed enclosed by supervisor")
    require(integer(done["peak_rss_bytes"], "worker RSS") <= 4 * 1024**3, "source-bound worker RSS cap")
    return terminal


def weighted_means(rows, weights, metrics):
    result = {}
    for arm in ARMS:
        strata = {hit: [row for row in rows if row["arm"] == arm and row["initial_hit"] == hit]
                  for hit in (1, 2, 3)}
        require(all(strata.values()), "all three strata for each arm")
        result[arm] = {metric: math.fsum(weights[hit] * math.fsum(row[metric] for row in strata[hit])
                                        / len(strata[hit]) for hit in (1, 2, 3)) for metric in metrics}
    return result


def candidate_conditions(means, blocks, candidate):
    """Prospective numeric comparisons; no epsilon is applied to any gate."""
    selected, full, exact = means[candidate], means["full_bayes"], means["exact_log"]
    differences = [block["means"]["recent32_hard"]["capped_time"] - block["means"][candidate]["capped_time"]
                   for block in blocks]
    return {
        "full_success_at_least_95pct": full["found"] >= .95,
        "no_failure_regression_vs_full": selected["found"] >= full["found"],
        "no_failure_regression_vs_exact": selected["found"] >= exact["found"],
        "time_at_most_105pct_full": selected["capped_time"] <= 1.05 * full["capped_time"],
        "time_at_most_105pct_exact": selected["capped_time"] <= 1.05 * exact["capped_time"],
        "strict_time_gain_vs_recent32": selected["capped_time"] < means["recent32"]["capped_time"],
        "strict_time_gain_vs_recent32_hard": selected["capped_time"] < means["recent32_hard"]["capped_time"],
        "positive_blocks_vs_recent32_hard_at_least_six": sum(value > 0 for value in differences) >= 6,
        "state_at_most_20pct_full": selected["state_array_bytes"] <= .20 * full["state_array_bytes"],
        "controller_at_most_150pct_full": selected["controller_seconds"] <= 1.5 * full["controller_seconds"],
    }


def utility_conditions(means, candidate):
    selected, full = means[candidate], means["full_bayes"]
    return {
        "no_failure_regression_vs_full": selected["found"] >= full["found"],
        "no_time_regression_vs_full": selected["capped_time"] <= full["capped_time"],
        "no_cost_regression_vs_full": selected["controller_seconds"] <= full["controller_seconds"],
        "strict_time_or_cost_gain_vs_full": selected["capped_time"] < full["capped_time"]
        or selected["controller_seconds"] < full["controller_seconds"],
    }


def episode(key, trace, row, comparison, budget):
    cohort, seed, arm = key
    case = seed - FIRST_SEEDS[cohort]
    hit, block = 1 + (case % 12) // 4, case // 12
    require((row["cohort"], row["seed"], row["arm"]) == key
            and row["initial_hit"] == hit and row["block"] == block, "episode identity")
    steps = integer(row["steps"], "episode steps", 1)
    require(steps <= HORIZON and row["capped_time"] == steps and type(row["found"]) is bool
            and (row["found"] or steps == HORIZON), "complete found or horizon-censored episode")
    require(len(trace) == steps + 1 and trace[0]["kind"] == "reset", "one reset and every primitive step")
    reset = trace[0]
    require(reset["block"] == block and reset["initial_hit"] == hit
            and reset["source_evaluation_only"] == row["source_evaluation_only"], "reset/episode join")
    source = coordinate(row["source_evaluation_only"])
    require(source != CENTER, "source excluded from initial visited cell")
    packet(reset["public"], 0, CENTER, hit)
    draws = iter(row["draw_log"])
    source_draw = next(draws, None)
    require(source_draw is not None, "one recorded source draw")
    source_uniform = draw_record(source_draw, "source", 0, source[0] * N + source[1], comparison)
    uniforms, position, positions = [], CENTER, [(0, 0), CENTER]
    repeated = stuck = 0
    maximum_tv = 0.0
    times = {name: [] for name in ("update_seconds", "decode_seconds", "planner_seconds", "environment_seconds")}
    for step, event in enumerate(trace[1:], 1):
        if step % 64 == 0:
            budget()
        require(event["kind"] == "step" and integer(event["step"], "primitive step", 1) == step,
                "contiguous primitive step chronology")
        action = integer(event["action"], "selected action")
        require(action == selected_action(event["scores"], position), "recorded first-within-tolerance action")
        target = moved(position, action)
        require(target != position, "legal nonblocked autonomous action")
        reading, found = packet(event["public"], step, target)
        require(found is (target == source) and event["found"] is found, "public found/source transition")
        if found:
            require(step == steps and row["found"], "no action after terminal source discovery")
        else:
            draw = next(draws, None)
            require(draw is not None, "nonterminal odor draw")
            uniforms.append(draw_record(draw, "hit", len(uniforms), reading, comparison))
        position = target
        repeated = repeated + 1 if position == positions[-2] else 0
        positions.append(position)
        require(type(event["stuck"]) is bool and event["stuck"] is (repeated > 8), "diagnostic stuck flag")
        stuck += event["stuck"]
        digest(event["posterior_before_sha256"], "recorded pre-action posterior witness")
        comparison.close(event["posterior_mass"], 1.0, "recorded normalized posterior mass", 1e-10)
        if arm in ("full_bayes", "exact_log"):
            error = finite(event["full_exact_tv_before"], "recorded own-path posterior parity", 0)
            require(error <= 1e-10, "recorded full/exact own-path qualification")
            maximum_tv = max(maximum_tv, error)
        else:
            require(event["full_exact_tv_before"] is None, "no compressed/recent exact-posterior claim")
        for name, values in times.items():
            values.append(finite(event[name], name, 0))
        if found:
            require(event["update_seconds"] == 0, "found sentinel never assimilated")
    require(next(draws, None) is None, "no extra draw or terminal odor sample")
    require(row["found"] is (position == source) and integer(row["stuck_steps"], "stuck total") == stuck,
            "terminal and stuck episode totals")
    for name, values in times.items():
        comparison.close(row[name], math.fsum(values), f"episode {name}")
    for name in ("actor_initialization_seconds", "environment_initialization_seconds"):
        finite(row[name], name, 0)
    finite(row["shared_initialization_allocation_seconds"], "amortized shared initialization", 0)
    finite(row["episode_seconds"], "whole episode elapsed", 0)
    comparison.close(row["controller_seconds"], math.fsum(row[name] for name in TIMES), "public controller total")
    expected_state = {"full_bayes": 53 * 53 * 9, "exact_log": 53 * 53 * 9,
                      "recent32": 53 * 53 + 32 * 3 * 8, "recent32_hard": 53 * 53 + 32 * 3 * 8,
                      "dct16_neutral": 16 * 16 * 8 + 53 * 53, "dct16_nearest": 16 * 16 * 8 + 53 * 53}[arm]
    require(integer(row["state_array_bytes"], "evolving actor arrays", 1) == expected_state,
            "declared evolving array payload including hard mask or observation ring")
    storage = row["storage"]
    if "state_array_bytes" in storage:
        require(storage["state_array_bytes"] == storage["evidence_array_bytes"] + storage["support_mask_bytes"] == expected_state,
                "additive/spectral storage arithmetic")
    else:
        require(storage["mutable_array_bytes"] == sum(storage["mutable_arrays"].values()) == expected_state,
                "baseline storage arithmetic")
    for field in ("update_calls", "decode_calls", "planner_calls"):
        integer(row[field], field)
    require(row["update_calls"] == steps - int(row["found"])
            and row["decode_calls"] == row["planner_calls"] == steps, "all observation/decision calls accounted")
    return {"source": source, "source_uniform": source_uniform,
            "source_probabilities": source_draw["probabilities"], "hit_uniforms": uniforms,
            "maximum_recorded_full_exact_tv": maximum_tv}


def matched_draws(record, shared, comparison):
    """Pair recorded channels, not observations produced at different positions."""
    if not shared:
        shared.update(record)
        return
    require(record["source"] == shared["source"] and record["source_uniform"] == shared["source_uniform"],
            "matched within-case source and uniform")
    comparison.same(record["source_probabilities"], shared["source_probabilities"], "paired source distribution")
    common = min(len(record["hit_uniforms"]), len(shared["hit_uniforms"]))
    require(record["hit_uniforms"][:common] == shared["hit_uniforms"][:common], "paired odor-channel uniform prefixes")
    if len(record["hit_uniforms"]) > len(shared["hit_uniforms"]):
        shared["hit_uniforms"] = record["hit_uniforms"]


def check_manifest(run, done, budget):
    require(run.is_dir() and not run.is_symlink(), "real completed run directory")
    require({path.name for path in run.iterdir()} == FILES | {"receipt.json"}
            and set(done["files"]) == FILES, "exact completed payload closure")
    total = 0
    for name, witness in done["files"].items():
        path = safe_file(run, name)
        require(path.stat().st_size == integer(witness["bytes"], f"payload bytes {name}")
                and sha(path) == digest(witness["sha256"], f"payload {name}"), f"payload bytes/hash {name}")
        total += path.stat().st_size
        budget()
    require(total + (run / "receipt.json").stat().st_size <= 512 * 1024**2, "complete recorded output within cap")


def source_manifest(root, sources, budget):
    require(isinstance(sources, dict) and sources, "nonempty source manifest")
    for name, pin in sources.items():
        require(sha(safe_file(root, name)) == digest(pin, f"source {name}"), f"source bytes {name}")
        budget()


def authenticate(args, budget):
    require(not args.run.is_symlink() and not args.plan.is_symlink() and not args.terminal.is_symlink(),
            "direct external input paths")
    run = args.run.resolve()
    require(sha(args.plan) == args.plan_sha256 and sha(run / "receipt.json") == args.receipt_sha256,
            "external plan and worker receipt pins")
    plan, done = read(args.plan), read(run / "receipt.json")
    require(plan["version"] == "otto-spectral-control-v1" and plan["status"] == "frozen_before_native_run",
            "prospective scientific plan")
    cohort_config = {name: {"first_seed": FIRST_SEEDS[name], "qualification_seed": 650001 + index * 10000,
                           "config": {"Ndim": 2, "lambda_over_dx": 3.0 + index, "R_dt": 2.0,
                                      "norm_Poisson": "Euclidean", "Ngrid": N, "Nhits": NHITS}}
                     for index, name in enumerate(COHORTS)}
    require(plan["configuration"] == {"arms": list(ARMS), "cohorts": cohort_config, "cases_per_cohort": 96,
            "horizon": HORIZON, "episodes": 1152, "rotation": "global_case_index modulo6", "score_tolerance": 1e-8,
            "posterior_tv_tolerance": 1e-10, "learned_pilot_admission": False}, "fixed two-regime cohort")
    require(plan["limits"] == done["limits"] == LIMITS and done["status"] == "completed"
            and done["phase"] == "cohort" and done["completed_episodes"] == 1152,
            "complete worker under original allocation")
    require(done["plan_sha256"] == args.plan_sha256 and done["sources"] == plan["sources"]
            and done["upstream_sources"] == plan["upstream_sources"], "worker/plan source joins")
    required = {"scripts/study_otto_spectral_control.py", "tests/test_otto_spectral_control.py",
                "research/otto-spectral-control-protocol.md"} | PINNED.keys()
    require(required <= plan["sources"].keys() and all(plan["sources"][name] == pin for name, pin in PINNED.items()),
            "fixed inherited code and required new source closure")
    require(plan["upstream_commit"] == "a6aaef6507cffd2aff79291c1019f506f616bbef", "frozen upstream revision")
    require({"LICENSE", "isotropic/classes/sourcetracking.py", "isotropic/classes/heuristicpolicy.py",
             "isotropic/classes/policy.py"} <= plan["upstream_sources"].keys(), "upstream execution source membership")
    source_manifest(ROOT, plan["sources"], budget)
    source_manifest(ROOT / "tmp/otto-source-review-01", plan["upstream_sources"], budget)
    check_manifest(run, done, budget)
    imports = read(run / "imports.json")
    require(imports["versions"] == plan["runtime_versions"] == {"numpy": "2.5.3", "scipy": "1.18.1"}
            and imports["python"].split()[0] == plan["python_version"] == "3.12.13"
            and imports["executable"] == plan["python_executable"] == str(ROOT / "tmp/otto-runtime-01/bin/python"),
            "recorded qualified runtime")
    expected_imports = {f"isotropic.classes.{name}": str(ROOT / f"tmp/otto-source-review-01/isotropic/classes/{name}.py")
                        for name in ("sourcetracking", "heuristicpolicy", "policy")}
    require(imports["upstream"] == expected_imports, "recorded normal upstream import paths")
    require(imports["numerical_thread_environment"] == {name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "recorded single-thread settings")
    terminal = authenticate_process(args, plan, done, imports)
    started = read(run / "started.json")
    require(started["request"] == {"plan": str(args.plan.resolve()), "plan_sha256": args.plan_sha256,
             "supervision": str(args.terminal.resolve()).replace(".terminal.json", ".launch.json"), "output": str(run)}
            and started["supervision_sha256"] == done["supervision_sha256"], "started request binding")
    require(started["clock_backend"] == done["clock_backend"] == terminal["clock_backend"]
            and started["deadline_ns"] == terminal["deadline_ns"]
            and terminal["started_ns"] <= started["started_ns"] == done["started_ns"]
            <= done["finished_ns"] <= terminal["finished_ns"]
            and done["elapsed_ns"] == done["finished_ns"] - done["started_ns"]
            and done["native_elapsed_seconds"] == done["elapsed_ns"] / 1e9, "nested worker native timestamps")
    require(done["training_updates"] == done["external_model_calls"] == 0
            and done["learned_pilot_admission"] is False, "no learned/model admission")
    require(integer(done["native_steps_attempted"], "native attempted") == done["native_steps_returned"]
            <= LIMITS["native_steps"], "all allocated native calls returned")
    return plan, done, terminal


def qualify(run, done, comparison, budget):
    qualified = read(run / "qualification.json")
    suffixes = ("explicit_geometry", "kernel_formula", "initial_hit_mixture", "replay_hit1", "global_rng_hit1",
                "replay_hit2", "global_rng_hit2", "replay_hit3", "global_rng_hit3", "saturated_boundary_found")
    names = [f"{cohort}.{suffix}" for cohort in COHORTS for suffix in suffixes]
    require(qualified["status"] == "completed" and [item["name"] for item in qualified["checks"]] == names
            and all(item["passes"] is True for item in qualified["checks"]), "all twenty structural witnesses")
    with (run / "qualification.jsonl").open() as stream:
        require([json.loads(line) for line in stream] == qualified["checks"], "incremental/completed qualification join")
    checks = {item["name"]: item for item in qualified["checks"]}
    weights, replay_steps = {}, 0
    for cohort in COHORTS:
        values = qualified["initial_hit_weights"][cohort]
        require(set(values) == {"1", "2", "3"}, "qualified three-stratum mixture")
        weights[cohort] = {int(hit): finite(value, "initial-hit weight", 0) for hit, value in values.items()}
        require(all(0 < value < 1 for value in values.values()), "positive nondegenerate initial-hit weights")
        comparison.close(math.fsum(values.values()), 1.0, "qualified mixture normalization", 1e-12)
        probabilities = checks[f"{cohort}.initial_hit_mixture"]["probabilities"]
        comparison.same(probabilities, [0.0, *(weights[cohort][hit] for hit in (1, 2, 3))], "qualification mixture join")
        for field in ("kernel_formula", "initial_hit_mixture"):
            require(0 <= finite(checks[f"{cohort}.{field}"]["maximum_absolute_difference"], "recorded analytic error") <= 1e-12,
                    "inherited sensor formula qualification tolerance")
        finite(qualified["shared_model_initialization_seconds"][cohort], "shared initialization", 0)
        for hit in (1, 2, 3):
            witness = checks[f"{cohort}.replay_hit{hit}"]
            steps = integer(witness["steps"], "qualification replay steps", 1)
            require(steps <= 84 and len(witness["replay"]) == steps, "bounded qualification replay")
            require(0 <= finite(witness["maximum_tv"], "recorded qualified TV") <= 1e-10
                    and 0 <= finite(witness["maximum_score_error"], "recorded qualified scores") <= 1e-8,
                    "inherited full/exact/full-rank qualification tolerances")
            source, position = coordinate(witness["source_evaluation_only"]), CENTER
            require(source != CENTER, "qualified source is not initial agent position")
            draws = iter(witness["draw_log"])
            draw_record(next(draws), "source", 0, source[0] * N + source[1], comparison)
            hit_index = 0
            for step, event in enumerate(witness["replay"], 1):
                require(event["step"] == step, "qualification replay chronology")
                position = moved(position, event["action"])
                reading, found = packet(event["public"], step, position)
                require(found is (position == source) and (not found or step == steps), "qualified terminal/source join")
                if not found:
                    draw_record(next(draws), "hit", hit_index, reading, comparison)
                    hit_index += 1
            require(position == source and next(draws, None) is None, "qualified replay source discovery and complete draw list")
            replay_steps += steps
            budget()
        require(checks[f"{cohort}.saturated_boundary_found"]["native_steps"] == 110,
                "recorded saturated/boundary fixture native work")
    expected_steps = 3 * replay_steps + 220
    require(qualified["native_steps"] == done["qualification_steps"] == expected_steps <= 2048,
            "qualification native work reconstructed from six replays and fixed fixtures")
    return qualified, weights


def decision_records(means, blocks, candidate):
    selected, full, exact = means[candidate], means["full_bayes"], means["exact_log"]
    independent = candidate_conditions(means, blocks, candidate)
    definitions = [
        ("full_success_at_least_95pct", full["found"], .95, ">=", "full_success_at_least_95pct"),
        ("no_failure_regression_vs_full_bayes", selected["found"] - full["found"], 0, ">=", "no_failure_regression_vs_full"),
        ("time_at_most_105pct_full_bayes", selected["capped_time"], 1.05 * full["capped_time"], "<=", "time_at_most_105pct_full"),
        ("no_failure_regression_vs_exact_log", selected["found"] - exact["found"], 0, ">=", "no_failure_regression_vs_exact"),
        ("time_at_most_105pct_exact_log", selected["capped_time"], 1.05 * exact["capped_time"], "<=", "time_at_most_105pct_exact"),
        ("strict_time_gain_vs_recent32", selected["capped_time"], means["recent32"]["capped_time"], "<", "strict_time_gain_vs_recent32"),
        ("strict_time_gain_vs_recent32_hard", selected["capped_time"], means["recent32_hard"]["capped_time"], "<", "strict_time_gain_vs_recent32_hard"),
    ]
    gains = [block["means"]["recent32_hard"]["capped_time"] - block["means"][candidate]["capped_time"] for block in blocks]
    definitions.extend([
        ("positive_blocks_vs_recent32_hard_at_least_six", sum(gain > 0 for gain in gains), 6, ">=", "positive_blocks_vs_recent32_hard_at_least_six"),
        ("state_at_most_20pct_full", selected["state_array_bytes"], .20 * full["state_array_bytes"], "<=", "state_at_most_20pct_full"),
        ("controller_at_most_150pct_full", selected["controller_seconds"], 1.5 * full["controller_seconds"], "<=", "controller_at_most_150pct_full"),
    ])
    rules = [{"name": name, "value": value, "threshold": threshold, "relation": relation, "passes": independent[key]}
             for name, value, threshold, relation, key in definitions]
    rules[7]["block_gains"] = gains
    utility = utility_conditions(means, candidate)
    utility_records = [
        {"name": "no_failure_regression_vs_full", "passes": utility["no_failure_regression_vs_full"], "value": selected["found"] - full["found"]},
        {"name": "time_no_worse_than_full", "passes": utility["no_time_regression_vs_full"], "value": selected["capped_time"] - full["capped_time"]},
        {"name": "controller_no_worse_than_full", "passes": utility["no_cost_regression_vs_full"], "value": selected["controller_seconds"] - full["controller_seconds"]},
        {"name": "time_or_controller_strictly_better", "passes": utility["strict_time_or_cost_gain_vs_full"]},
    ]
    return rules, utility_records


def summarize(rows, weights):
    result = {}
    for cohort in COHORTS:
        selected = [row for row in rows if row["cohort"] == cohort]
        require(len(selected) == 576, "complete per-regime cohort")
        means = weighted_means(selected, weights[cohort], METRICS)
        strata = {str(hit): {arm: {metric: math.fsum(row[metric] for row in selected
                  if row["arm"] == arm and row["initial_hit"] == hit) / 32 for metric in METRICS}
                  for arm in ARMS} for hit in (1, 2, 3)}
        blocks = [{"block": block, "means": weighted_means([row for row in selected if row["block"] == block],
                    weights[cohort], METRICS)} for block in range(8)]
        decisions = {candidate: decision_records(means, blocks, candidate) for candidate in CANDIDATES}
        result[cohort] = {"initial_hit_weights": {str(hit): value for hit, value in weights[cohort].items()},
                          "means": means, "strata": strata, "blocks": blocks,
                          "criteria_by_candidate": {candidate: decisions[candidate][0] for candidate in CANDIDATES},
                          "utility_criteria_by_candidate": {candidate: decisions[candidate][1] for candidate in CANDIDATES},
                          "compact_by_candidate": {candidate: all(rule["passes"] for rule in decisions[candidate][0]) for candidate in CANDIDATES},
                          "utility_by_candidate": {candidate: all(rule["passes"] for rule in decisions[candidate][1]) for candidate in CANDIDATES}}
    return {"episodes": len(rows), "cohorts": result, "structural_qualification": True,
            "compact_control_viable": all(result[cohort]["compact_by_candidate"][candidate] for cohort in COHORTS for candidate in CANDIDATES),
            "utility_compute_advantage": all(result[cohort]["utility_by_candidate"][candidate] for cohort in COHORTS for candidate in CANDIDATES),
            "utility_by_candidate": {candidate: all(result[cohort]["utility_by_candidate"][candidate] for cohort in COHORTS) for candidate in CANDIDATES},
            "learned_pilot_admission": False, "inherited_gate_revised": False}


def pair_rows(rows, weights):
    indexed = {(row["cohort"], row["seed"], row["arm"]): row for row in rows}
    for cohort in COHORTS:
        for case in range(96):
            seed = FIRST_SEEDS[cohort] + case
            hit = 1 + (case % 12) // 4
            for candidate in CANDIDATES:
                selected = indexed[cohort, seed, candidate]
                for control in ARMS[:4]:
                    reference = indexed[cohort, seed, control]
                    yield {"cohort": cohort, "seed": seed, "block": case // 12, "initial_hit": hit,
                           "candidate": candidate, "control": control, "case_mixture_weight": weights[cohort][hit] / 32,
                           "capped_time_gain": reference["capped_time"] - selected["capped_time"],
                           "success_difference": int(selected["found"]) - int(reference["found"]),
                           "controller_seconds_gain": reference["controller_seconds"] - selected["controller_seconds"],
                           "candidate_steps": selected["steps"], "control_steps": reference["steps"],
                           "candidate_found": selected["found"], "control_found": reference["found"]}


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    comparison = Comparison()
    start = last = None
    progress = {"episodes": 0, "cohort_native_steps": 0}
    own_pin = sha(__file__)
    try:
        require(own_pin == args.source_sha256, "external auditor source pin")
        require(sha(ROOT / "src/openjev/research/suspend_clock.py") == CLOCK_PIN, "audit clock source")
        clock = SuspendClock()
        start = clock.now_ns()
        deadline = start + 300 * 10**9

        def budget():
            nonlocal last
            last = clock.now_ns()
            require(last < deadline, "saved audit deadline")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                    <= 2 * 1024**3, "saved audit RSS cap")
            require(sum(path.stat().st_size for path in output.iterdir() if path.is_file()) <= 64 * 1024**2,
                    "saved audit output cap")

        write(output / "started.json", {"scope": SCOPE, "source_sha256": own_pin,
              "request": {key: str(value) for key, value in vars(args).items()}, "clock_backend": clock.backend,
              "started_ns": start, "deadline_ns": deadline})
        plan, done, terminal = authenticate(args, budget)
        run = args.run.resolve()
        qualification, weights = qualify(run, done, comparison, budget)
        episode_groups, trace_groups = grouped(run / "episodes.jsonl"), grouped(run / "transitions.jsonl")
        rows, maximum_tv = [], 0.0
        for regime_index, cohort in enumerate(COHORTS):
            for case in range(96):
                paired = {}
                shift = (regime_index * 96 + case) % 6
                for arm in ARMS[shift:] + ARMS[:shift]:
                    budget()
                    key = (cohort, FIRST_SEEDS[cohort] + case, arm)
                    ep, trace = next(episode_groups, None), next(trace_groups, None)
                    require(ep is not None and trace is not None and ep[0] == trace[0] == key and len(ep[1]) == 1,
                            "all 1152 episodes and traces exactly once in fixed order")
                    row = ep[1][0]
                    evidence = episode(key, trace[1], row, comparison, budget)
                    matched_draws(evidence, paired, comparison)
                    maximum_tv = max(maximum_tv, evidence["maximum_recorded_full_exact_tv"])
                    shared = qualification["shared_model_initialization_seconds"][cohort] / 96 if arm in (
                        "exact_log", *CANDIDATES) else 0.0
                    comparison.close(row["shared_initialization_allocation_seconds"], shared, "shared model setup allocation", 0)
                    rows.append(row)
                    progress["episodes"] += 1
                    progress["cohort_native_steps"] += row["steps"]
        require(next(episode_groups, None) is None and next(trace_groups, None) is None, "no extra episode or trace")
        require(progress["episodes"] == 1152 and done["native_steps_returned"]
                == qualification["native_steps"] + progress["cohort_native_steps"], "all native work and cohort membership")
        recomputed = summarize(rows, weights)
        producer = read(run / "summary.json")
        comparison.same({key: producer[key] for key in recomputed}, recomputed, "complete scientific summary")
        comparison.close(producer["full_exact_maximum_tv"], maximum_tv, "maximum recorded own-path TV", 0)
        comparison.close(done["full_exact_maximum_tv"], maximum_tv, "receipt maximum own-path TV", 0)
        comparison.same(producer["shared_model_initialization_seconds"], qualification["shared_model_initialization_seconds"],
                        "shared setup qualification/summary join")
        comparison.close(producer["sum_controller_seconds"], math.fsum(row["controller_seconds"] for row in rows), "total logical controller cost")
        comparison.close(producer["sum_measured_actor_seconds"], math.fsum(row["controller_seconds"]
                         - row["shared_initialization_allocation_seconds"] for row in rows), "total measured actor cost")
        comparison.close(producer["sum_environment_seconds"], math.fsum(row["environment_seconds"]
                         + row["environment_initialization_seconds"] for row in rows), "total environment cost")
        for name in ("compact_control_viable", "utility_compute_advantage", "learned_pilot_admission"):
            comparison.same(done[name], recomputed[name], f"worker final {name}")
        pairs = list(pair_rows(rows, weights))
        require(len(pairs) == 1536, "all candidate/reference matched episode differences")
        with (output / "paired-episodes.jsonl").open("x") as stream:
            for row in pairs:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        summary = {"agreement": True, "scope": SCOPE, "progress": progress,
                   "qualification_native_steps": qualification["native_steps"], "total_native_steps": done["native_steps_returned"],
                   "scalar_checks": comparison.scalars, "maximum_absolute_difference": comparison.maximum_error,
                   "candidate_condition_checks": 40, "utility_condition_checks": 16, "paired_episode_contrasts": len(pairs),
                   "recomputed_summary": recomputed,
                   "cost_scope": "Recorded component arithmetic and logical shared-model allocation are checked. Timing/RSS truth, shared-array allocation truth and simulator internals are not independently measured.",
                   "random_scope": "Recorded source distributions, inverse-CDF choices and matched source/hit uniform channels are checked. The PRNG and physical likelihood are not regenerated."}
        write(output / "summary.json", summary)
        require(sha(__file__) == own_pin and sha(args.plan) == args.plan_sha256
                and sha(run / "receipt.json") == args.receipt_sha256
                and sha(args.terminal) == args.terminal_sha256, "unchanged external bindings")
        source_manifest(ROOT, plan["sources"], budget)
        source_manifest(ROOT / "tmp/otto-source-review-01", plan["upstream_sources"], budget)
        check_manifest(run, done, budget)
        budget()
        receipt = {"status": "completed", "agreement": True, "scope": SCOPE, "source_sha256": own_pin,
                   "plan_sha256": args.plan_sha256, "producer_receipt_sha256": args.receipt_sha256,
                   "producer_summary_sha256": done["files"]["summary.json"]["sha256"], "terminal_sha256": args.terminal_sha256,
                   "progress": progress, "scalar_checks": comparison.scalars,
                   "maximum_absolute_difference": comparison.maximum_error, "environment_calls": 0, "model_calls": 0,
                   "parent_original_wall_seconds": terminal["wall_seconds"], "clock_backend": clock.backend,
                   "started_ns": start, "finished_ns": last, "elapsed_ns": last - start, "wall_seconds": (last - start) / 1e9,
                   "files": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                             for path in output.iterdir() if path.is_file()}}
        write(output / "receipt.json", receipt)
        budget()
        print(json.dumps({"status": "completed", "agreement": True, **progress,
                          "scalar_checks": comparison.scalars}), flush=True)
        return receipt
    except BaseException as error:
        try:
            if (output / "receipt.json").exists():
                (output / "receipt.json").rename(output / "invalid-receipt.json")
            write(output / "failed.json", {"status": "failed", "source_sha256": own_pin, "scope": SCOPE,
                  "error": repr(error), "traceback": traceback.format_exc(), "progress": progress,
                  "last_clock_elapsed_ns": None if last is None or start is None else last - start,
                  "timing_available": False, "wall_seconds": None})
        except BaseException as secondary:  # noqa: BLE001 - preserve the audit's primary failure.
            error.add_note(f"Failure evidence publication also failed: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "plan", "terminal", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("plan-sha256", "receipt-sha256", "terminal-sha256", "source-sha256"):
        parser.add_argument(f"--{flag}", required=True)
    execute(parser.parse_args())
