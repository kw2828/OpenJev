"""Independent saved-output verification of the fixed OTTO memory screen.

No runner, adapter, actor, upstream simulator or policy imports. NumPy/SciPy
reconstruct the public likelihood model, categorical draws, beliefs and one-step
policy scores. This verifies saved calculations, not simulator execution truth,
planner optimality, learned efficacy or the official expected-time benchmark.
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

import numpy as np
from scipy.special import k0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

ARMS = ("space_full", "space_recent32", "space_recent8", "space_initial",
        "info_full", "info_recent32", "info_recent8", "info_initial")
METRICS = ("capped_time", "found", "controller_seconds", "initialization_seconds",
           "filter_seconds", "planning_seconds", "environment_seconds", "stuck_steps")
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
COMMIT = "a6aaef6507cffd2aff79291c1019f506f616bbef"
N, HORIZON, EPS = 19, 642, 1e-10
QUAL_NAMES = ["official_autoset_geometry", "all_likelihoods_include_tail_and_zero_origin", "initial_hit_mixture",
              "initial_prior_and_source_hit1", "seeded_and_forced_hit_replay_hit1", "global_numpy_rng_unchanged_hit1",
              "initial_prior_and_source_hit2", "seeded_and_forced_hit_replay_hit2", "global_numpy_rng_unchanged_hit2",
              "full_public_filter_and_policy_parity", "saturated_hit_public_update", "boundary_action_stays_put",
              "source_found_terminal", "qualification_native_step_budget"]
FILES = {"imports.json", "qualification.json", "public-kernel.npz", "transitions.jsonl", "episodes.jsonl", "summary.json"}
FILES |= {f"qualification-{i:02}.json" for i in range(1, 15)}
SOURCES = {"src/openjev/research/otto_public.py", "src/openjev/research/otto_memory.py",
           "scripts/study_otto_memory.py", "tests/test_otto_public.py", "tests/test_otto_memory.py",
           "tests/test_otto_memory_study.py", "research/otto-memory-protocol.md", "research/otto-runtime-requirements.txt",
           "scripts/supervise_dialogue_observation_v2.py", "src/openjev/research/suspend_clock.py",
           "output/otto-memory-v1/engineering-01/receipt.json"}
SOURCES |= {f"output/otto-memory-v1/engineering-01/check-{i:02}.log" for i in range(3)}
UPSTREAM_SOURCES = {"LICENSE", "requirements.txt", "README.md", "isotropic/classes/sourcetracking.py",
                    "isotropic/classes/gymwrapper.py", "isotropic/classes/heuristicpolicy.py",
                    "isotropic/classes/rlpolicy.py", "isotropic/classes/policy.py", "isotropic/evaluate/evaluate.py",
                    "isotropic/evaluate/parameters/isotropic-19x19.py", "isotropic/evaluate/parameters/__defaults.py"}
SCOPE = ("Independent saved source/process/payload bindings, 512 rotated cases, public movement/found/visited "
         "semantics, analytic Poisson kernel and initial mixture, PCG64 uniforms and categorical CDF choices, "
         "public belief and all four one-step action scores, recorded costs, eight weighted blocks and ten gates. "
         "Posterior byte hashes are bound witnesses, not claimed bitwise reproduction. Actual simulator/policy "
         "execution, qualification's forced-reference objects, isolation, timing/RSS truth and planner optimality "
         "remain inherited. No native environment, model or training calls.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"integer: {label}")
    return value


def finite(value, label, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"finite: {label}")
    require(minimum is None or value >= minimum, f"minimum: {label}")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class Comparison:
    def __init__(self):
        self.scalars = 0
        self.maximum_error = 0.0
        self.maximum_score_error = 0.0

    def close(self, actual, expected, label, tolerance=1e-10):
        finite(actual, label)
        error = abs(actual - expected)
        self.scalars += 1
        self.maximum_error = max(self.maximum_error, error)
        require(error <= tolerance, f"numeric disagreement: {label}: {actual!r} versus {expected!r}")

    def same(self, actual, expected, label):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), f"keys: {label}")
            for key, value in expected.items():
                self.same(actual[key], value, f"{label}.{key}")
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"length: {label}")
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.same(left, right, f"{label}[{index}]")
        elif type(expected) is float:
            self.close(actual, expected, label)
        else:
            self.scalars += 1
            require(type(actual) is type(expected) and actual == expected, f"exact: {label}")


def authenticate(args, budget):
    run, plan_path, terminal_path = args.run.resolve(), args.plan.resolve(), args.terminal.resolve()
    require(not args.run.is_symlink(), "real run directory")
    require(sha(plan_path) == args.plan_sha256, "external plan pin")
    require(sha(run / "receipt.json") == args.receipt_sha256, "external receipt pin")
    require(sha(terminal_path) == args.terminal_sha256, "external terminal pin")
    plan, done, terminal = read(plan_path), read(run / "receipt.json"), read(terminal_path)
    require(done["status"] == terminal["status"] == "completed", "complete worker and actual parent required")
    require({p.name for p in run.iterdir()} == FILES | {"receipt.json"} and set(done["files"]) == FILES,
            "exact run closure without late/failure/partial artifacts")
    for name, witness in done["files"].items():
        path = run / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == witness["bytes"]
                and sha(path) == witness["sha256"], f"payload: {name}")
        budget()
    require(sum(p.stat().st_size for p in run.iterdir()) <= 256 * 1024**2, "worker output cap")
    require(done["plan_sha256"] == args.plan_sha256 and done["sources"] == plan["sources"]
            and done["upstream_sources"] == plan["upstream_sources"], "plan and source joins")
    require(set(plan["sources"]) == SOURCES and set(plan["upstream_sources"]) == UPSTREAM_SOURCES
            and plan["upstream_commit"] == COMMIT, "exact fourteen local and eleven upstream source closure")
    require(plan["status"] == "frozen_before_native_execution" and plan["cohort"] == {
        "arms": list(ARMS), "seeds": list(range(510001, 510065)), "episodes": 512, "horizon": HORIZON,
        "weighting": "exact upstream initial-hit mixture, balanced 32 cases in each positive initial-hit stratum",
        "case_assignment": "case=seed-510001, block=case//8, hit=1+(case%8)//4; rotate arm order left bycase%8"}, "fixed cohort")
    require(plan["limits"] == {"native_seconds": 1800, "rss_bytes": 4 * 1024**3, "output_bytes": 256 * 1024**2,
                               "native_steps": 340000, "qualification_native_steps": 1000}, "fixed resource limits")
    require(plan["native_models"] == plan["training_updates"] == plan["external_models"] == 0, "no model allocation")
    require(plan["engineering_receipt"] == "output/otto-memory-v1/engineering-01/receipt.json"
            and plan["engineering_sha256"] == plan["sources"][plan["engineering_receipt"]], "engineering lineage")
    for root, sources in ((ROOT, plan["sources"]), (ROOT / "tmp/otto-source-review-01", plan["upstream_sources"])):
        for name, pin in sources.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "safe source path")
            require(not (root / name).is_symlink() and sha(root / name) == pin, f"source: {name}")
    require(plan["sources"]["src/openjev/research/suspend_clock.py"] == CLOCK_PIN, "clock pin")
    imports = read(run / "imports.json")
    require(imports["versions"] == plan["runtime_versions"] == {"numpy": "2.5.3", "scipy": "1.18.1"}
            and imports["python"].split()[0] == plan["python_version"] == "3.12.13"
            and imports["executable"] == plan["python_executable"] == str(ROOT / "tmp/otto-runtime-01/bin/python"),
            "recorded runtime versions")
    require(set(imports["upstream"]) == {f"isotropic.classes.{x}" for x in ("sourcetracking", "heuristicpolicy", "policy")},
            "normal upstream import inventory")
    for name, path in imports["upstream"].items():
        relative = name.replace(".", "/") + ".py"
        require(Path(path) == ROOT / "tmp/otto-source-review-01" / relative
                and relative in plan["upstream_sources"], "imported upstream path and source pin")
    launch_path = terminal_path.with_name(terminal_path.name.replace(".terminal.json", ".launch.json"))
    require(launch_path != terminal_path and sha(launch_path) == done["supervision_sha256"], "launch pin")
    launch = read(launch_path)
    command = list(launch["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:2] == [imports["executable"], str(ROOT / "scripts/study_otto_memory.py")], "worker command")
    require(len(command[2:]) == 8, "four command arguments")
    flags = dict(zip(command[2::2], command[3::2], strict=True))
    require(flags == {"--output": str(run), "--supervision": str(launch_path), "--plan": str(plan_path),
                      "--plan-sha256": args.plan_sha256}, "exact worker flags")
    for key in ("version", "command", "cwd", "clock_backend", "started_ns", "deadline_ns", "cap_seconds", "pid", "pgid",
                "parent_pid", "watchdog_sha256", "clock_source_sha256"):
        require(launch[key] == terminal[key], f"parent join: {key}")
    require(launch["version"] == "dialogue-observation-supervision-v2" and Path(launch["cwd"]).resolve() == ROOT,
            "parent version and cwd")
    require(launch["clock_backend"] in {"mach_continuous_time", "CLOCK_BOOTTIME"}
            and launch["clock_source_sha256"] == CLOCK_PIN
            and launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"], "parent clock/source")
    start, finish = integer(launch["started_ns"], "start"), integer(terminal["finished_ns"], "finish")
    require(launch["cap_seconds"] == 1800 and launch["deadline_ns"] == start + 1800 * 10**9
            and start <= finish < launch["deadline_ns"] and terminal["elapsed_ns"] == finish - start
            and terminal["wall_seconds"] == (finish - start) / 1e9, "strict native parent deadline")
    require(terminal["timing_available"] is True and terminal["returncode"] == 0 and terminal["group_absent"] is True
            and terminal["timed_out"] is False and terminal["error"] is None and terminal["clock_error"] is None,
            "actual successful terminal")
    require(terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["errors"] == [], "cleanup success")
    require(integer(launch["pid"], "pid", 1) == launch["pgid"] != launch["parent_pid"], "isolated worker")
    for stamp in (launch["started_unix"], terminal["started_unix"], terminal["finished_unix"]):
        finite(stamp, "civil provenance", 0)
    require(0 <= finite(done["native_elapsed_seconds"], "worker elapsed") <= terminal["wall_seconds"], "worker elapsed enclosure")
    require(integer(done["peak_rss_bytes"], "worker RSS") <= 4 * 1024**3, "worker RSS cap")
    require(done["training_updates"] == done["external_model_calls"] == 0 and done["phase"] == "cohort", "no trained/model route")
    budget()
    return plan, done, terminal


def normalized(p):
    mass = float(np.sum(p))
    require(np.isfinite(p).all() and np.all(p >= 0) and math.isfinite(mass) and mass > 0, "positive finite posterior")
    return p / mass


def poisson_categories(mu):
    zero = np.exp(-mu)
    one = mu * zero
    # The third category is the entire >=2 tail, not the exact count two.
    return np.stack((zero, one, np.maximum(0.0, 1.0 - zero - one)))


def analytic_kernel():
    coordinates = np.indices((39, 39)) - 19
    radius = np.sqrt(np.sum(coordinates**2, axis=0))
    result = poisson_categories(k0(np.where(radius == 0, 1, radius)) / math.log(2))
    result[:, 19, 19] = 0
    r = np.arange(1, 1000, dtype=np.float64)
    likelihood = poisson_categories(k0(r) / math.log(2))
    shell = 2 * math.pi * r
    mass = [math.fsum((likelihood[h] * shell).tolist()) for h in (1, 2)]
    return result, np.array([0.0, mass[0], mass[1]]) / math.fsum(mass)


def likelihood_at(kernel, position):
    x, y = position
    return kernel[:, N - x:2 * N - x, N - y:2 * N - y]


def prior(kernel, hit):
    p = np.full((N, N), 1.0 / (N * N - 1))
    p[9, 9] = 0
    return normalized(p * likelihood_at(kernel, (9, 9))[hit])


def moved(position, action):
    result = list(position)
    axis, delta = action // 2, -1 if action % 2 == 0 else 1
    result[axis] = min(N - 1, max(0, result[axis] + delta))
    return tuple(result)


def packet(value, step, position, hit, done):
    require(set(value) == {"position", "hit", "done", "step", "valid_actions"}, "public field whitelist")
    require(value["position"] == list(position) and value["step"] == step and value["hit"] == hit
            and type(value["done"]) is bool and value["done"] is done, "public packet values")
    require(all(type(v) is int for v in value["position"]) and type(value["step"]) is int
            and type(value["hit"]) is int, "public integer types")
    expected = [] if done else [a for a in range(4) if moved(position, a) != position]
    require(value["valid_actions"] == expected, "public valid-action set")


def entropy(p):
    keep = p > EPS
    return float(np.sum(-p[keep] * np.log2(p[keep])))


def action_scores(p, position, kernel, space):
    result = [None] * 4
    grid = np.indices((N, N))
    base_entropy = entropy(p)
    for action in range(4):
        target = moved(position, action)
        if target == position:
            continue
        end_mass = float(p[target])
        if end_mass > 1 - EPS:
            value = -EPS
        else:
            surviving = p.copy()
            surviving[target] = 0
            if surviving.sum() > EPS:
                surviving /= surviving.sum()
            branches = surviving[None, ...] * likelihood_at(kernel, target)
            hit_mass = np.sum(branches, axis=(1, 2))
            positive = hit_mass > EPS
            branches[positive] /= hit_mass[positive, None, None]
            entropies = [entropy(branch) for branch in branches]
            if space:
                distance = np.abs(grid[0] - target[0]) + np.abs(grid[1] - target[1])
                values = []
                for branch, h in zip(branches, entropies, strict=True):
                    estimate = float(np.sum(branch * distance)) + 2**(h - 1) - .5
                    values.append(math.log2(estimate) if estimate > 0 else estimate)
                value = (1 - end_mass) * math.fsum(float(w) * v for w, v in zip(hit_mass, values, strict=True))
            else:
                value = (1 - end_mass) * math.fsum(float(w) * h for w, h in zip(hit_mass, entropies, strict=True))
        result[action] = value if space else base_entropy - value
    return result


class Draws:
    def __init__(self, seed, records, comparison):
        self.records = iter(records)
        self.rng = {name: np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index])))
                    for index, name in enumerate(("initial", "source", "hit"))}
        self.counts = dict.fromkeys(self.rng, 0)
        self.comparison = comparison

    def choice(self, channel, expected, selected):
        row = next(self.records, None)
        require(row is not None and set(row) == {"channel", "index", "probabilities", "cdf_mass", "uniform", "selected_index"}, "draw fields")
        require(row["channel"] == channel and row["index"] == self.counts[channel], "draw channel order")
        actual = np.asarray(row["probabilities"], dtype=np.float64)
        require(actual.shape == expected.shape and np.isfinite(actual).all() and np.all(actual >= 0), "draw probability support")
        require(abs(float(actual.sum()) - 1) <= 1e-10, "draw normalized mass")
        self.comparison.close(float(np.max(np.abs(actual - expected))), 0.0, "analytic draw probability", 1e-12)
        cumulative = np.cumsum(actual)
        self.comparison.close(row["cdf_mass"], float(cumulative[-1]), "draw cumulative mass", 0.0)
        uniform = float(self.rng[channel].random())
        self.comparison.close(row["uniform"], uniform, "seeded channel uniform", 0.0)
        index = int(np.searchsorted(cumulative / cumulative[-1], uniform, side="right"))
        require(type(row["selected_index"]) is int and row["selected_index"] == index == selected, "categorical inverse CDF")
        self.counts[channel] += 1

    def finish(self):
        require(next(self.records, None) is None, "no extra or terminal hit draw")


def source_position(value):
    require(isinstance(value, list) and len(value) == 2
            and all(type(x) is int and 0 <= x < N for x in value), "evaluation source coordinates")
    require(value != [9, 9], "source excludes initial visited cell")
    return tuple(value)


def qualify(run, kernel, weights, comparison):
    qualification = read(run / "qualification.json")
    require(qualification["status"] == "completed" and [r["name"] for r in qualification["checks"]] == QUAL_NAMES,
            "all fourteen structural witnesses")
    checks = {r["name"]: r for r in qualification["checks"]}
    for i, row in enumerate(qualification["checks"], 1):
        require(row["passes"] is True and read(run / f"qualification-{i:02}.json") == row, "individual qualification witness")
    comparison.same(checks["initial_hit_mixture"]["weights"], weights.tolist(), "qualified initial mixture")
    for name in ("all_likelihoods_include_tail_and_zero_origin", "initial_hit_mixture"):
        require(0 <= checks[name]["maximum_absolute_difference"] <= 1e-12, "qualification analytic tolerance")
    require(0 <= checks["full_public_filter_and_policy_parity"]["maximum_absolute_difference"] <= EPS,
            "qualification public filter tolerance")
    total = 0
    for hit in (1, 2):
        trace = [r for r in qualification["replay"] if r["initial_hit"] == hit]
        witness = checks[f"seeded_and_forced_hit_replay_hit{hit}"]
        require(len(trace) == witness["steps"] and trace, "qualification replay length")
        source = source_position(witness["source"])
        draws = Draws(410010 + hit, witness["draw_log"], comparison)
        draws.choice("source", prior(kernel, hit).ravel(), source[0] * N + source[1])
        position = (9, 9)
        for t, row in enumerate(trace, 1):
            require(row["step"] == t and row["exact_replay"] is True, "qualification contiguous recorded replay")
            action = integer(row["action"], "qualification action")
            require(action < 4, "qualification action range")
            position = moved(position, action)
            found = position == source
            require(row["position"] == list(position) and row["done"] is found
                    and (row["hit"] == -2 if found else type(row["hit"]) is int and 0 <= row["hit"] < 3), "qualification public semantics")
            if found:
                require(t == len(trace), "qualification stop at found")
            else:
                draws.choice("hit", likelihood_at(kernel, position)[:, source[0], source[1]], row["hit"])
        require(position == source, "qualification reaches source")
        draws.finish()
        total += len(trace)
    require(checks["full_public_filter_and_policy_parity"]["transitions_checked"] == total, "qualification filter coverage")
    # Three step calls per recorded replay, then 1 saturated +12 north +1 blocked +27 to (18,18).
    expected_native = 3 * total + 41
    require(qualification["native_steps"] == checks["qualification_native_step_budget"]["steps"] == expected_native <= 1000,
            "qualification attempted/returned work")
    return qualification


def grouped(path):
    with path.open() as stream:
        rows = (json.loads(line) for line in stream)
        for key, values in itertools.groupby(rows, key=lambda r: (r["seed"], r["arm"])):
            yield key, list(values)


def episode(key, trace, row, kernel, comparison, budget):
    seed, arm = key
    case = seed - 510001
    hit, block = 1 + (case % 8) // 4, case // 8
    require(row["seed"] == seed and row["arm"] == arm and row["initial_hit"] == hit and row["block"] == block, "episode identity")
    steps = integer(row["steps"], "episode steps", 1)
    require(steps <= HORIZON and row["capped_time"] == steps and type(row["found"]) is bool
            and (row["found"] or steps == HORIZON), "full capped episode")
    require(len(trace) == steps + 1 and trace[0]["kind"] == "reset", "one reset and all transitions")
    reset = trace[0]
    require(reset["block"] == block and reset["initial_hit"] == hit
            and reset["source_evaluation_only"] == row["source_evaluation_only"], "reset joins")
    source = source_position(row["source_evaluation_only"])
    position = (9, 9)
    packet(reset["public"], 0, position, hit, False)
    initial = prior(kernel, hit)
    belief = initial.copy()
    draws = Draws(seed, row["draw_log"], comparison)
    draws.choice("source", initial.ravel(), source[0] * N + source[1])
    window = {"full": None, "recent32": 32, "recent8": 8, "initial": 0}[arm.split("_")[1]]
    visited, history = {(9, 9)}, []
    positions = [(0, 0), position]
    repeated = stuck = 0
    maximum_filter_error = 0.0
    times = {k: [] for k in ("planning_seconds", "filter_seconds", "environment_seconds")}
    space = arm.startswith("space")
    for step, event in enumerate(trace[1:], 1):
        if step % 16 == 0:
            budget()
        require(event["kind"] == "step" and event["step"] == step, "contiguous primitive steps")
        action = integer(event["action"], "action")
        require(action < 4 and moved(position, action) != position, "legal nonblocked cohort action")
        expected_scores = action_scores(belief, position, kernel, space)
        scores = event["scores"]
        require(isinstance(scores, list) and len(scores) == 4, "four action scores")
        for actual, expected in zip(scores, expected_scores, strict=True):
            if expected is None:
                require(actual is None, "invalid action score mask")
            else:
                finite(actual, "recorded action score")
                difference = abs(actual - expected)
                comparison.maximum_score_error = max(comparison.maximum_score_error, difference)
                comparison.close(actual, expected, "independent one-step action score", 1e-8)
        valid = [a for a, score in enumerate(scores) if score is not None]
        best = min(scores[a] for a in valid) if space else max(scores[a] for a in valid)
        chosen = next(a for a in valid if abs(scores[a] - best) < EPS)
        require(action == chosen, "first-within-1e-10 action tie rule")
        position = moved(position, action)
        found = position == source
        reading = event["public"]["hit"]
        require(reading == -2 if found else type(reading) is int and 0 <= reading < 3, "found sentinel or categorical reading")
        packet(event["public"], step, position, reading, found)
        if found:
            require(step == steps and row["found"], "stop at actual source")
            belief = np.zeros((N, N))
            belief[source] = 1
        else:
            draws.choice("hit", likelihood_at(kernel, position)[:, source[0], source[1]], reading)
            visited.add(position)
            history.append((position, reading))
            if window is None:
                belief[position] = 0
                belief = normalized(belief * likelihood_at(kernel, position)[reading])
            else:
                belief = initial.copy()
                for cell in visited:
                    belief[cell] = 0
                belief = normalized(belief)
                for cell, value in history[-window:] if window else []:
                    belief = normalized(belief * likelihood_at(kernel, cell)[value])
            require(all(belief[cell] == 0 for cell in visited), "permanent visited-cell exclusion")
        comparison.close(event["posterior_mass"], 1.0, "recorded posterior normalization")
        digest = event["posterior_sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "posterior hash witness")
        if window is None:
            error = finite(event["full_filter_max_error"], "recorded filter parity", 0)
            require(error <= EPS, "recorded full filter tolerance")
            maximum_filter_error = max(maximum_filter_error, error)
        else:
            require(event["full_filter_max_error"] is None, "no partial-memory native parity claim")
        repeated = repeated + 1 if position == positions[-2] else 0
        positions.append(position)
        require(type(event["stuck"]) is bool and event["stuck"] is (repeated > 8), "back-and-forth stuck counter")
        stuck += event["stuck"]
        for name, values in times.items():
            values.append(finite(event[name], name, 0))
    require(row["found"] is (position == source) and row["stuck_steps"] == stuck, "episode terminal and stuck totals")
    draws.finish()
    for name, values in times.items():
        comparison.close(row[name], math.fsum(values), f"episode {name}", 1e-9)
    finite(row["initialization_seconds"], "initialization", 0)
    comparison.close(row["controller_seconds"], math.fsum(row[k] for k in ("initialization_seconds", "planning_seconds", "filter_seconds")),
                     "initialization plus public inference cost", 1e-9)
    return maximum_filter_error


def summarize(rows, weights):
    def means(subset):
        answer = {}
        for arm in ARMS:
            strata = {h: [r for r in subset if r["arm"] == arm and r["initial_hit"] == h] for h in (1, 2)}
            require(all(strata.values()), "both initial-hit strata")
            answer[arm] = {m: math.fsum(weights[h] * math.fsum(r[m] for r in strata[h]) / len(strata[h]) for h in (1, 2))
                           for m in METRICS}
        return answer
    overall = means(rows)
    strata = {str(h): {arm: {m: math.fsum(r[m] for r in rows if r["initial_hit"] == h and r["arm"] == arm) / 32
                            for m in METRICS} for arm in ARMS} for h in (1, 2)}
    blocks = [{"block": b, "means": means([r for r in rows if r["block"] == b])} for b in range(8)]
    full, recent = overall["space_full"], overall["space_recent32"]
    gain = recent["capped_time"] - full["capped_time"]
    block_gains = [b["means"]["space_recent32"]["capped_time"] - b["means"]["space_full"]["capped_time"] for b in blocks]
    wins = sum(g > 0 for g in block_gains)
    rules = [
        {"name": "full_success_at_least_95pct", "value": full["found"], "passes": full["found"] >= .95},
        {"name": "gain_vs_recent32_at_least_10pct", "value": gain / recent["capped_time"], "passes": gain / recent["capped_time"] >= .10},
        {"name": "gain_vs_recent32_at_least_two_steps", "value": gain, "passes": gain >= 2},
        {"name": "positive_blocks_vs_recent32_at_least_six", "value": wins, "block_gains": block_gains, "passes": wins >= 6},
    ]
    for arm in ("space_recent32", "space_recent8"):
        difference = full["found"] - overall[arm]["found"]
        rules.append({"name": f"no_failure_regression_vs_{arm}", "value": difference, "passes": difference >= 0})
    for arm in ("space_recent8", "space_initial"):
        difference = overall[arm]["capped_time"] - full["capped_time"]
        rules.append({"name": f"strict_time_gain_vs_{arm}", "value": difference, "passes": difference > 0})
    rules.extend(({"name": "complete_cohort", "value": 512, "passes": True},
                  {"name": "structural_qualification", "value": True, "passes": True}))
    return {"scope": "classical odor-memory opportunity screen; no trained architecture or official evaluator replication",
            "episodes": 512, "initial_hit_weights": {str(h): weights[h] for h in (1, 2)}, "means": overall,
            "strata": strata, "blocks": blocks, "criteria": rules,
            "learned_pilot_opportunity": all(r["passes"] for r in rules), "structural_qualification_passed": True}


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    own_pin, comparison = sha(__file__), Comparison()
    start = last = None
    try:
        require(sha(ROOT / "src/openjev/research/suspend_clock.py") == CLOCK_PIN, "audit clock source")
        clock = SuspendClock()
        start = clock.now_ns()
        deadline = start + 300 * 10**9

        def budget():
            nonlocal last
            last = clock.now_ns()
            require(last < deadline, "audit deadline")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                    <= 2 * 1024**3, "audit RSS cap")
            require(sum(p.stat().st_size for p in output.iterdir()) <= 256 * 1024**2, "audit output cap")

        write(output / "started.json", {"source_sha256": own_pin, "request": {k: str(v) for k, v in vars(args).items()},
                                       "clock_backend": clock.backend, "started_ns": start, "deadline_ns": deadline, "scope": SCOPE})
        plan, done, terminal = authenticate(args, budget)
        run = args.run.resolve()
        with np.load(run / "public-kernel.npz", allow_pickle=False) as saved:
            require(set(saved.files) == {"likelihood", "initial_hit_weights"}, "kernel NPZ keys")
            kernel, weights = saved["likelihood"], saved["initial_hit_weights"]
            require(kernel.dtype == weights.dtype == np.dtype("float64") and kernel.shape == (3, 39, 39)
                    and weights.shape == (3,), "kernel shape and precision")
            require(np.isfinite(kernel).all() and np.all(kernel >= 0) and np.isfinite(weights).all(), "kernel finite support")
        analytic, analytic_weights = analytic_kernel()
        comparison.close(float(np.max(np.abs(kernel - analytic))), 0.0, "all 4563 kernel cells", 1e-12)
        comparison.close(float(np.max(np.abs(weights - analytic_weights))), 0.0, "initial annular mixture", 1e-12)
        require(weights[0] == 0 and all(0 < w < 1 for w in weights[1:]) and abs(float(weights.sum()) - 1) <= 1e-12,
                "positive two-stratum weights")
        qualification = qualify(run, kernel, weights, comparison)
        ep_groups, tr_groups = grouped(run / "episodes.jsonl"), grouped(run / "transitions.jsonl")
        rows, maximum_filter_error = [], 0.0
        for case in range(64):
            source = None
            shift = case % 8
            for arm in ARMS[shift:] + ARMS[:shift]:
                budget()
                key = (510001 + case, arm)
                eg, tg = next(ep_groups, None), next(tr_groups, None)
                require(eg is not None and tg is not None and eg[0] == tg[0] == key and len(eg[1]) == 1,
                        "all 512 cases exactly once in fixed rotation")
                row = eg[1][0]
                error = episode(key, tg[1], row, kernel, comparison, budget)
                maximum_filter_error = max(maximum_filter_error, error)
                if source is None:
                    source = row["source_evaluation_only"]
                require(row["source_evaluation_only"] == source, "same conditional source across all arms")
                rows.append(row)
        require(next(ep_groups, None) is None and next(tr_groups, None) is None, "no trailing episodes/transitions")
        steps = sum(r["steps"] for r in rows)
        require(done["native_steps_attempted"] == done["native_steps_returned"] == steps + qualification["native_steps"] <= 340000,
                "complete qualification plus cohort native-call accounting")
        require(done["qualification_steps"] == qualification["native_steps"] and done["completed_episodes"] == 512,
                "qualification/episode root counters")
        require(math.fsum(r["controller_seconds"] + r["environment_seconds"] for r in rows)
                <= done["native_elapsed_seconds"] + 1e-9, "disjoint cohort timing within whole worker time")
        require(done["active_case"] == {"seed": rows[-1]["seed"], "arm": rows[-1]["arm"]}, "last active case")
        result = summarize(rows, {h: float(weights[h]) for h in (1, 2)})
        result["full_filter_max_error"] = maximum_filter_error
        comparison.same(read(run / "summary.json"), result, "complete producer summary")
        require(done["learned_pilot_opportunity"] is result["learned_pilot_opportunity"], "root opportunity decision")
        for name, witness in done["files"].items():
            require(sha(run / name) == witness["sha256"], "payload unchanged during audit")
            budget()
        for name, pin in plan["sources"].items():
            require(sha(ROOT / name) == pin, "frozen source unchanged during audit")
        require(sha(args.plan) == args.plan_sha256 and sha(run / "receipt.json") == args.receipt_sha256
                and sha(args.terminal) == args.terminal_sha256, "external bindings unchanged during audit")
        require(sha(__file__) == own_pin, "audit source unchanged")
        budget()
        write(output / "summary.json", {"agreement": True, "scope": SCOPE, "episodes": 512, "cohort_steps": steps,
              "qualification_steps": qualification["native_steps"], "native_calls": 0, "model_calls": 0,
              "scalar_comparisons": comparison.scalars, "maximum_absolute_difference": comparison.maximum_error,
              "maximum_action_score_difference": comparison.maximum_score_error, "recomputed_summary": result})
        budget()
        receipt = {"status": "completed", "agreement": True, "source_sha256": own_pin,
                   "plan_sha256": args.plan_sha256, "producer_receipt_sha256": args.receipt_sha256,
                   "producer_summary_sha256": done["files"]["summary.json"]["sha256"], "terminal_sha256": args.terminal_sha256,
                   "parent_wall_seconds": terminal["wall_seconds"], "clock_backend": clock.backend,
                   "started_ns": start, "finished_ns": last, "elapsed_ns": last - start, "wall_seconds": (last - start) / 1e9,
                   "limits": {"native_seconds": 300, "rss_bytes": 2 * 1024**3, "output_bytes": 256 * 1024**2},
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                   "scalar_comparisons": comparison.scalars, "maximum_absolute_difference": comparison.maximum_error,
                   "learned_pilot_opportunity": result["learned_pilot_opportunity"], "scope": SCOPE,
                   "timing_scope": "Native start through last pre-receipt check; publication rechecked before return.",
                   "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}}
        write(output / "receipt.json", receipt)
        budget()
        print(json.dumps({"status": "completed", "agreement": True, "episodes": 512,
                          "learned_pilot_opportunity": result["learned_pilot_opportunity"]}), flush=True)
    except BaseException as exc:
        try:
            if (output / "receipt.json").exists():
                (output / "receipt.json").rename(output / "invalid-receipt.json")
            write(output / "failed.json", {"status": "failed", "agreement": False, "error": repr(exc),
                  "traceback": traceback.format_exc(), "source_sha256": own_pin,
                  "last_clock_elapsed_ns": None if last is None else last - start, "timing_available": False,
                  "wall_seconds": None, "scalar_comparisons_before_failure": comparison.scalars})
        except BaseException as publication_error:  # noqa: BLE001 - preserve original failure
            if hasattr(exc, "add_note"):
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
