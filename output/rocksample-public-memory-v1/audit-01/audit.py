"""Independent saved-trace coverage and score arithmetic; no filter/env replay.

Only authentication imports the qualified native clock helper. Forecast values
are inherited; matching pending forecasts and past-only age metadata does not
independently prove producer chronology or Bayesian posterior correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("prior", "full", "recent32", "recent128", "latest_check")
MAPS = tuple(range(11001, 11009))
RESETS = tuple(range(21001, 21005))
AGES = ("never", "le32", "33to128", "gt128")
CAPS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
SOURCE = "scripts/diagnose_rocksample_public_memory.py"
SCOPE = (
    "Independent raw observation/action structure, complete membership, pending-forecast joins, "
    "past-only age metadata, binary NLL/Brier, all map/age aggregates and six decisions. "
    "Forecast probabilities, actual RNG policy sampling, simulator dynamics/hidden state, "
    "chronology and native timing truth are inherited from pinned producer/qualification. "
    "No Bayesian probability recomputation, environment calls, models or training."
)


def require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pinned(path, pin):
    require(sha(path) == pin, f"SHA mismatch: {path}")
    return read(path)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class Budget:
    def __init__(self, out):
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research.suspend_clock import SuspendClock
        self.clock = SuspendClock()
        self.deadline = self.clock.deadline_after(CAPS["wall_seconds"])
        self.out = out

    def check(self):
        self.deadline.check()
        require(peak_rss() <= CAPS["rss_bytes"], "audit RSS cap")
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= CAPS["output_bytes"], "audit output cap")

    def timing(self):
        elapsed = self.deadline.elapsed_ns()
        return {"clock_backend": self.clock.backend, "wall_seconds": elapsed / 1e9,
                "elapsed_ns": elapsed, "timing_available": True}


class Comparison:
    def __init__(self):
        self.scalar_checks = 0
        self.max_error = 0.

    def __call__(self, actual, expected, path="value"):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), f"keys: {path}")
            for key in expected:
                self(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), f"length: {path}")
            for index, value in enumerate(expected):
                self(actual[index], value, f"{path}[{index}]")
        else:
            self.scalar_checks += 1
            if type(expected) is float:
                require(type(actual) in (int, float) and math.isfinite(actual), f"finite scalar: {path}")
                error = abs(actual - expected)
                self.max_error = max(error, self.max_error)
                require(error <= 1e-12, f"numerical discrepancy at {path}: {actual!r} versus {expected!r}")
            else:
                require(type(actual) is type(expected) and actual == expected, f"exact discrepancy at {path}")


def authenticate(args, budget):
    plan = pinned(args.plan, args.plan_sha256)
    require(plan["status"] == "frozen_before_any_diagnostic_calls", "plan status")
    require(plan["collection"] == {"maps": list(MAPS), "resets_per_map": list(RESETS),
            "steps_per_fragment": 256, "fragments": 32, "transitions": 8192}, "fixed collection")
    for name, pin in plan["source_hashes"].items():
        require(sha(ROOT / name) == pin, f"frozen source changed: {name}")
    done = pinned(args.run / "receipt.json", args.receipt_sha256)
    terminal = pinned(args.terminal, args.terminal_sha256)
    require(done["status"] == "completed" and done["transitions"] == 8192 and done["completed_fragments"] == 32, "incomplete collection")
    require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False,
            "successful actual terminal required")
    require(terminal["timing_available"] is True and terminal["clock_error"] is None and terminal["error"] is None,
            "untrusted native terminal")
    require(terminal["group_absent"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["errors"] == [], "supervisor cleanup")
    require(terminal["clock_backend"] in {"mach_continuous_time", "CLOCK_BOOTTIME"}, "native backend")
    require(terminal["deadline_ns"] == terminal["started_ns"] + 600 * 10**9
            and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
            and 0 < terminal["elapsed_ns"] < 600 * 10**9
            and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9, "terminal interval")
    launch_path = Path(done["supervision"]).resolve()
    launch = pinned(launch_path, done["supervision_sha256"])
    for key in ("command", "cwd", "parent_pid", "pid", "pgid", "cap_seconds", "clock_backend", "started_ns", "deadline_ns", "version"):
        require(launch[key] == terminal[key], f"launch/terminal join: {key}")
    require(launch["version"] == "dialogue-observation-supervision-v2" and launch["cap_seconds"] == 600,
            "supervisor recipe")
    require(launch["pid"] == launch["pgid"] and launch["pid"] > 0, "isolated worker group")
    command = list(launch["command"])
    if command[1] == "-u":
        command.pop(1)
    require(Path(command[1]).resolve() == ROOT / SOURCE and command[2:] == ["--output", str(args.run), "--supervision", str(launch_path)], "actual command")
    require(Path(launch["cwd"]).resolve() == ROOT, "actual cwd")
    require(done["native_started_ns"] == launch["started_ns"] and 0 < done["native_elapsed_seconds"] <= terminal["wall_seconds"], "producer interval")
    require(done["peak_rss_bytes"] <= plan["limits"]["peak_rss_bytes"], "producer RSS")
    require(terminal["watchdog_sha256"] == plan["source_hashes"]["scripts/supervise_dialogue_observation_v2.py"]
            and terminal["clock_source_sha256"] == plan["source_hashes"]["src/openjev/research/suspend_clock.py"], "supervisor sources")
    expected_source_names = {SOURCE, "src/openjev/research/rocksample_public_belief.py",
        "research/rocksample-public-memory-diagnostic-protocol.md", "scripts/qualify_rocksample_runtime.py",
        "scripts/supervise_dialogue_observation_v2.py", "src/openjev/research/suspend_clock.py",
        "research/rocksample-runtime-requirements.lock.txt"}
    require(done["source_hashes"] == {name: plan["source_hashes"][name] for name in expected_source_names}, "producer source closure")
    require(done["qualified_runtime_sha256"] == plan["runtime_qualification_sha256"], "qualified runtime pin")
    qualification = ROOT / "output/rocksample-runtime-v1/qualification-01"
    qualified = pinned(qualification / "completed.json", plan["runtime_qualification_sha256"])
    require(qualified["status"] == "completed", "qualification incomplete")
    for name, descriptor in qualified["files"].items():
        require(sha(qualification / name) == descriptor["sha256"], f"qualified payload: {name}")
    require(set(done["output_hashes"]) == {"traces.jsonl", "predictions.jsonl", "summary.json"}, "three diagnostic payloads")
    require({p.name for p in args.run.iterdir()} == set(done["output_hashes"]) | {"receipt.json"}, "exact complete run membership")
    for name, pin in done["output_hashes"].items():
        require(sha(args.run / name) == pin, f"diagnostic payload hash: {name}")
    require(done["output_hashes"]["summary.json"] == args.summary_sha256, "external summary pin")
    require(sum(p.stat().st_size for p in args.run.iterdir()) <= plan["limits"]["output_bytes"], "producer bytes")
    budget.check()
    return plan, done, terminal


def observation(row):
    values = row["observation"]
    require(isinstance(values, list) and len(values) == 33 and all(type(v) in (int, float) and math.isfinite(v) for v in values), "raw observation shape/finite")
    for part in (values[:11], values[11:22]):
        require(all(v in (0, 1) for v in part) and sum(part) == 1, "raw one-hot coordinate")
    require(all(v in (-1, 0, 1) for v in values[22:]), "signed rock channels")
    coordinate = (values[:11].index(1), values[11:22].index(1))
    require(coordinate[1] < 10, "collector must avoid terminal east column")
    return coordinate, values[22:]


def recompute_losses(probabilities, positive):
    require(set(probabilities) == set(ARMS), "forecast arm membership")
    result = {}
    for arm in ARMS:
        probability = probabilities[arm]
        require(type(probability) in (int, float) and math.isfinite(probability) and 0 <= probability <= 1, "probability bounds")
        likelihood = probability if positive else 1 - probability
        require(likelihood > 0, "impossible event must not be silently clipped")
        difference = Fraction(probability) - int(positive)
        result[arm] = {"nll": -math.log(likelihood), "brier": float(difference * difference)}
    return result


def lines(path):
    with path.open() as stream:
        for text in stream:
            require(bool(text.strip()), "blank ledger row")
            yield json.loads(text)


def verify_rows(args, compare, budget):
    trace = iter(lines(args.run / "traces.jsonl"))
    scored = iter(lines(args.run / "predictions.jsonl"))
    records = []
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    action_counts = [0] * 16
    for map_seed in MAPS:
        for reset_seed in RESETS:
            initial = next(trace)
            require(set(initial) == {"map_seed", "reset_seed", "step", "observation"}, "reset fields")
            require((initial["map_seed"], initial["reset_seed"], initial["step"]) == (map_seed, reset_seed, -1), "reset identity/order")
            previous, readings = observation(initial)
            require(not any(readings), "reset reading channels")
            check_positions = [[] for _ in range(11)]
            samples = []
            for step in range(256):
                row = next(trace)
                require(set(row) == {"map_seed", "reset_seed", "step", "action", "observation", "done", "pending_forecast"}, "transition fields")
                require((row["map_seed"], row["reset_seed"], row["step"]) == (map_seed, reset_seed, step), "transition identity/order")
                require(row["done"] is False, "unexpected boundary")
                action = row["action"]
                require(type(action) is int and 0 <= action < 16, "action range")
                action_counts[action] += 1
                coordinate, readings = observation(row)
                if action < 4:
                    require(not (action == 1 and previous[1] == 9), "collector east redirection")
                    expected = tuple(max(0, min(10, v + dv)) for v, dv in zip(previous, moves[action], strict=True))
                else:
                    expected = previous
                require(coordinate == expected, "public action/coordinate transition")
                if action >= 5:
                    rock = action - 5
                    require(readings[rock] in (-1, 1) and sum(v != 0 for v in readings) == 1, "selected query/observation alignment")
                    past = check_positions[rock]
                    latest = past[-1] if past else None
                    lag = step - latest if latest is not None else None
                    age = "never" if lag is None else "le32" if lag <= 32 else "33to128" if lag <= 128 else "gt128"
                    probabilities = row["pending_forecast"]
                    require(isinstance(probabilities, dict) and probabilities["prior"] == .5, "fixed prior/pending forecast")
                    positive = readings[rock] == 1
                    expected_record = {"map_seed": map_seed, "reset_seed": reset_seed, "step": step,
                        "rock": rock, "age": age, "prior_reading_age": lag, "prior_checks": len(past),
                        "samples_since_last_check": sum(t > latest for t in samples) if latest is not None else len(samples),
                        "predictions": probabilities, "positive": positive, "losses": recompute_losses(probabilities, positive)}
                    actual = next(scored)
                    require(actual["predictions"] == probabilities, "pending and scored forecasts must be identical")
                    compare(actual, expected_record, f"endpoint[{len(records)}]")
                    records.append(expected_record)
                    past.append(step)
                else:
                    require(not any(readings) and row["pending_forecast"] is None, "non-check channels/forecast")
                    if action == 4:
                        samples.append(step)
                previous = coordinate
            budget.check()
    require(next(trace, None) is None and next(scored, None) is None, "unexpected extra ledger rows")
    require(sum(action_counts) == 8192 and sum(action_counts[5:]) == len(records), "action/endpoint coverage")
    return records, action_counts


def average(rows):
    if not rows:
        return None
    return {arm: {metric: math.fsum(r["losses"][arm][metric] for r in rows) / len(rows)
                  for metric in ("nll", "brier")} for arm in ARMS}


def summarize(records):
    maps = []
    for seed in MAPS:
        selected = [row for row in records if row["map_seed"] == seed]
        require(bool(selected), "every map requires checks")
        ages = {}
        for age in AGES:
            subset = [row for row in selected if row["age"] == age]
            ages[age] = {"count": len(subset), "scores": average(subset)}
        maps.append({"map_seed": seed, "checks": len(selected), "scores": average(selected), "ages": ages})
    means = {arm: {metric: math.fsum(row["scores"][arm][metric] for row in maps) / 8
                   for metric in ("nll", "brier")} for arm in ARMS}
    criteria = []
    for control in ("recent128", "latest_check"):
        differences = [row["scores"][control]["nll"] - row["scores"]["full"]["nll"] for row in maps]
        require(means[control]["nll"] > 0, "relative-gain denominator")
        criteria += [{"name": f"relative_nll_gain_vs_{control}",
            "value": 1 - means["full"]["nll"] / means[control]["nll"],
            "passes": means["full"]["nll"] <= .9 * means[control]["nll"]},
            {"name": f"positive_maps_vs_{control}", "value": sum(value > 0 for value in differences),
             "passes": sum(value > 0 for value in differences) >= 6, "map_gains": differences}]
    delayed = [row for row in records if row["age"] == "gt128"]
    delayed_maps = [row["ages"]["gt128"]["scores"] for row in maps if row["ages"]["gt128"]["count"]]
    criteria += [{"name": "delayed_endpoints", "value": len(delayed), "passes": len(delayed) >= 32},
                 {"name": "delayed_maps", "value": len(delayed_maps), "passes": len(delayed_maps) >= 4}]
    delayed_mean = {arm: {metric: math.fsum(row[arm][metric] for row in delayed_maps) / len(delayed_maps)
                         for metric in ("nll", "brier")} for arm in ARMS} if delayed_maps else None
    age_breakdown = {}
    for age in AGES:
        subset = [row for row in records if row["age"] == age]
        age_breakdown[age] = {"count": len(subset), "scores": average(subset)}
    return {"scope": "fixed exploratory prediction diagnostic; no training or control comparison",
            "scores_equal_map_means": means, "by_map": maps, "check_endpoints": len(records),
            "delayed_equal_map_means": delayed_mean, "delayed_contributing_maps": len(delayed_maps),
            "pooled_age_breakdown": age_breakdown, "criteria": criteria,
            "memory_pilot_admitted": all(row["passes"] for row in criteria)}


def fixtures():
    values = recompute_losses(dict.fromkeys(ARMS, .5), True)
    require(values["full"] == {"nll": math.log(2), "brier": .25}, "fair-coin fixture")
    require(recompute_losses(dict.fromkeys(ARMS, 1.), True)["full"] == {"nll": -0., "brier": 0.}, "certain-correct fixture")
    try:
        recompute_losses(dict.fromkeys(ARMS, 1.), False)
    except AssertionError:
        pass
    else:
        raise AssertionError("impossible-event fixture")
    require(average([]) is None, "empty mean fixture")


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    budget = None
    compare = Comparison()
    source = sha(__file__)
    try:
        budget = Budget(args.out)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("audit awake alarm")))
        signal.setitimer(signal.ITIMER_REAL, CAPS["wall_seconds"])
        write(args.out / "started.json", {"status": "started", "source_sha256": source,
              "request": {key: str(value) for key, value in vars(args).items()}, "limits": CAPS, "scope": SCOPE})
        fixtures()
        plan, done, terminal = authenticate(args, budget)
        records, actions = verify_rows(args, compare, budget)
        expected = summarize(records)
        compare(read(args.run / "summary.json"), expected, "summary")
        require(done["check_endpoints"] == len(records) and done["memory_pilot_admitted"] is expected["memory_pilot_admitted"], "receipt summary join")
        result = {"status": "completed", "agreement": True, "scope": SCOPE,
                  "fragments": 32, "transitions": 8192, "check_endpoints": len(records),
                  "action_counts": actions, "scalar_checks": compare.scalar_checks,
                  "maximum_absolute_difference": compare.max_error, "fixtures_passed": 4,
                  "recomputed_summary": expected, "model_calls": 0, "environment_calls": 0,
                  "producer_parent_wall_seconds": terminal["wall_seconds"]}
        write(args.out / "summary.json", result)
        require(sha(__file__) == source, "audit source changed")
        for name, pin in plan["source_hashes"].items():
            require(sha(ROOT / name) == pin, f"frozen source end stability: {name}")
        require(sha(args.run / "receipt.json") == args.receipt_sha256 and sha(args.terminal) == args.terminal_sha256
                and sha(args.plan) == args.plan_sha256, "pinned inputs changed")
        for name, pin in done["output_hashes"].items():
            require(sha(args.run / name) == pin, f"saved input changed: {name}")
        budget.check()
        write(args.out / "receipt.json", {"status": "completed", "agreement": True,
              "source_sha256": source, "scope": SCOPE,
              "producer_receipt_sha256": args.receipt_sha256, "producer_summary_sha256": args.summary_sha256,
              "plan_sha256": args.plan_sha256, "terminal_sha256": args.terminal_sha256,
              "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(args.out.iterdir())},
              "scalar_checks": compare.scalar_checks, "maximum_absolute_difference": compare.max_error,
              "memory_pilot_admitted": expected["memory_pilot_admitted"],
              "peak_rss_bytes": peak_rss(), **budget.timing()})
        budget.check()
        print(json.dumps({"status": "completed", "scalar_checks": compare.scalar_checks,
                          "receipt_sha256": sha(args.out / "receipt.json")}))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "late-receipt.json")
            write(args.out / "failed.json", {"status": "failed", "error": repr(error),
                  "source_sha256": source, "scalar_checks_before_failure": compare.scalar_checks,
                  "scope": SCOPE, "peak_rss_bytes": peak_rss()})
        except BaseException as secondary:  # noqa: BLE001 - retain the original cause
            error.add_note(f"Failure publication also failed: {secondary!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--run", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--terminal", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--out", type=lambda value: Path(value).resolve(), required=True)
    execute(parser.parse_args())
