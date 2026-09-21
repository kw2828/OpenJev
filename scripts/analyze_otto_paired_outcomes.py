"""Post hoc decomposition of authenticated 53x53 OTTO primary paired outcomes.

Saved artifacts only, using the Python standard library. No simulator, policy,
runner, auditor, model, hidden-source coordinates, fitting, or gate revision.
The original 9/10 continuation failure remains the decision for this cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PIN = "0719638892f11224fe63cb88bf52f254fbb2ab413091aa7160246142a378cd61"
ARMS = ("space_full", "space_recent32", "space_recent8", "space_initial",
        "info_full", "info_recent32", "info_recent8", "info_initial")
PRIMARY = ("space_full", "space_recent32")
STRATA = (1, 2, 3)
SEEDS = tuple(range(610001, 610097))
PAYLOADS = {"imports.json", "qualification.json", "public-kernel.npz",
            "transitions.jsonl", "episodes.jsonl", "summary.json"}
PAYLOADS |= {f"qualification-{i:02}.json" for i in range(1, 18)}
SCOPE = ("Post hoc description of all 96 space_full/space_recent32 pairs in the "
         "completed fixed 53x53 cohort. Original mixture weights retained. "
         "Concentration and omission estimates are descriptive, not new gates, "
         "significance tests, model results, or permission to extend this cohort.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def pinned(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"real input file: {path}")
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in "0123456789abcdef" for c in expected), "SHA-256 pin format")
    require(sha(path) == expected, f"input pin: {path}")


def authenticate(args):
    require(args.run.is_dir() and not args.run.is_symlink(), "real run directory")
    run, terminal_path = args.run.resolve(), args.terminal.resolve()
    pinned(run / "receipt.json", args.receipt_sha256)
    pinned(args.terminal, args.terminal_sha256)
    pinned(run / "summary.json", args.summary_sha256)
    receipt, terminal = read(run / "receipt.json"), read(terminal_path)
    require(receipt["status"] == terminal["status"] == "completed"
            and terminal["returncode"] == 0 and terminal["group_absent"] is True
            and terminal["timed_out"] is False and terminal["error"] is None
            and terminal["clock_error"] is None, "successful worker and parent required")
    require(receipt["plan_sha256"] == PLAN_PIN, "original frozen study identity")
    require(set(receipt["files"]) == PAYLOADS
            and {p.name for p in run.iterdir()} == PAYLOADS | {"receipt.json"},
            "exact completed payload closure")
    for name, witness in receipt["files"].items():
        path = run / name
        pinned(path, witness["sha256"])
        require(type(witness["bytes"]) is int and path.stat().st_size == witness["bytes"],
                f"payload length: {name}")
    require(receipt["files"]["summary.json"]["sha256"] == args.summary_sha256,
            "independent summary pin agrees with producer manifest")
    require(terminal_path.name.endswith(".terminal.json"), "terminal filename")
    launch_path = terminal_path.with_name(terminal_path.name.replace(".terminal.json", ".launch.json"))
    pinned(launch_path, receipt["supervision_sha256"])
    launch = read(launch_path)
    for key in ("version", "command", "cwd", "clock_backend", "started_ns", "deadline_ns",
                "cap_seconds", "pid", "pgid", "parent_pid", "watchdog_sha256", "clock_source_sha256"):
        require(launch[key] == terminal[key], f"parent launch binding: {key}")
    command = list(launch["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(len(command) == 10 and command[1] == str(ROOT / "scripts/study_otto_large_memory.py"),
            "original worker command")
    flags = dict(zip(command[2::2], command[3::2], strict=True))
    require(set(flags) == {"--plan", "--plan-sha256", "--supervision", "--output"}
            and flags["--plan-sha256"] == PLAN_PIN
            and Path(flags["--output"]).resolve() == run
            and Path(flags["--supervision"]).resolve() == launch_path
            and Path(launch["cwd"]).resolve() == ROOT, "worker input and output binding")
    plan_path = Path(flags["--plan"])
    pinned(plan_path, PLAN_PIN)
    require(terminal["timing_available"] is True and terminal["cap_seconds"] == 1800
            and terminal["started_ns"] <= terminal["finished_ns"] < terminal["deadline_ns"]
            and terminal["deadline_ns"] == terminal["started_ns"] + 1800 * 10**9
            and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
            and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9, "actual parent timing")
    require(terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["errors"] == [], "successful parent cleanup")
    require(receipt["completed_episodes"] == 768
            and receipt["training_updates"] == receipt["external_model_calls"] == 0,
            "complete original non-neural cohort")
    summary = read(run / "summary.json")
    require(summary["episodes"] == 768 and summary["learned_pilot_opportunity"] is False
            and len(summary["criteria"]) == 10
            and sum(c["passes"] is True for c in summary["criteria"]) == 9
            and [c["name"] for c in summary["criteria"] if c["passes"] is False]
            == ["positive_blocks_vs_recent32_at_least_six"], "unchanged original 9/10 failure")
    weights = {int(k): v for k, v in summary["initial_hit_weights"].items()}
    require(set(weights) == set(STRATA)
            and all(type(w) in (int, float) and math.isfinite(w) and 0 < w < 1 for w in weights.values())
            and abs(math.fsum(weights.values()) - 1) <= 1e-12, "three-stratum mixture")
    inputs = {"run": str(run), "producer_receipt_sha256": args.receipt_sha256,
              "producer_summary_sha256": args.summary_sha256, "terminal": str(terminal_path),
              "terminal_sha256": args.terminal_sha256, "launch": str(launch_path),
              "launch_sha256": receipt["supervision_sha256"], "plan": str(plan_path),
              "plan_sha256": PLAN_PIN, "payloads": receipt["files"]}
    return run, receipt, summary, weights, inputs


def pairs_from_saved(run, receipt, weights):
    episodes = {}
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    seen = set()
    with (run / "episodes.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["seed"], row["arm"]
            require(key in expected and key not in seen, "unique original episode")
            seen.add(key)
            if key[1] in PRIMARY:
                # Deliberately discard evaluator-only source and RNG fields.
                episodes[key] = {k: row[k] for k in ("steps", "capped_time", "found", "initial_hit", "block")}
    require(seen == expected, "all 768 original episodes present")
    traces, resets = defaultdict(list), {}
    native_steps = 0
    with (run / "transitions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["seed"], row["arm"]
            require(key in expected and row["kind"] in {"reset", "step"}, "known trace identity")
            if row["kind"] == "step":
                native_steps += 1
            if key[1] not in PRIMARY:
                continue
            if row["kind"] == "reset":
                require(key not in resets and not traces[key], "one initial reset")
                resets[key] = row["public"]
            else:
                require(key in resets, "reset before transitions")
                obs = row["public"]
                require(row["step"] == obs["step"] == len(traces[key]) + 1, "trace chronology")
                traces[key].append((row["action"], tuple(obs["position"]), obs["hit"], obs["done"]))
    require(native_steps + receipt["qualification_steps"] == receipt["native_steps_attempted"]
            == receipt["native_steps_returned"], "all native-step accounting")
    result = []
    for seed in SEEDS:
        case, pair = seed - SEEDS[0], {}
        for arm in PRIMARY:
            key = seed, arm
            episode, reset, trace = episodes[key], resets[key], traces[key]
            require(episode["initial_hit"] == 1 + (case % 12) // 4 and episode["block"] == case // 12,
                    "original stratum and block assignment")
            require(reset["step"] == 0 and reset["done"] is False
                    and reset["position"] == [26, 26] and reset["hit"] == episode["initial_hit"],
                    "original public reset")
            require(type(episode["steps"]) is int and 1 <= len(trace) == episode["steps"]
                    == episode["capped_time"] <= 2188 and episode["found"] is True
                    and trace[-1][3] is True and trace[-1][2] == -2
                    and all(x[3] is False and x[2] in (0, 1, 2, 3) for x in trace[:-1]),
                    "complete successful public trajectory")
            pair[arm] = episode["steps"]
        left, right = (traces[seed, arm] for arm in PRIMARY)
        first_action = next((i for i, (a, b) in enumerate(zip(left, right), 1) if a[0] != b[0]), None)
        first_public = next((i for i, (a, b) in enumerate(zip(left, right), 1) if a != b), None)
        require(first_action == first_public and (first_action is not None or left == right),
                "paired public paths share their prefix until the first action difference")
        hit = 1 + (case % 12) // 4
        gain = pair["space_recent32"] - pair["space_full"]
        result.append({"seed": seed, "initial_hit": hit, "block": case // 12,
                       "full_steps": pair["space_full"], "recent32_steps": pair["space_recent32"],
                       "gain_steps": gain, "weighted_contribution": weights[hit] * gain / 32,
                       "outcome": "win" if gain > 0 else "loss" if gain < 0 else "tie",
                       "first_action_difference": first_action, "first_public_difference": first_public,
                       "public_trajectories_identical": left == right})
    return result


def weighted_mean(rows, metric, weights):
    groups = {h: [r[metric] for r in rows if r["initial_hit"] == h] for h in weights}
    require(all(groups.values()), "cannot omit an entire initial-hit stratum")
    return math.fsum(weights[h] * math.fsum(values) / len(values) for h, values in groups.items())


def counts(rows):
    require(bool(rows), "nonempty descriptive group")
    return {"pairs": len(rows), "wins": sum(r["gain_steps"] > 0 for r in rows),
            "ties": sum(r["gain_steps"] == 0 for r in rows), "losses": sum(r["gain_steps"] < 0 for r in rows),
            "unweighted_mean_gain": math.fsum(r["gain_steps"] for r in rows) / len(rows),
            "contribution_to_original_weighted_mean": math.fsum(r["weighted_contribution"] for r in rows)}


def omission_estimate(rows, weights):
    gain = weighted_mean(rows, "gain_steps", weights)
    control = weighted_mean(rows, "recent32_steps", weights)
    return {"remaining_pairs": len(rows), "mean_gain_steps": gain,
            "recent32_mean_steps": control, "relative_gain": gain / control}


def analyze(pairs, weights, original):
    gain = weighted_mean(pairs, "gain_steps", weights)
    for arm, metric in (("space_full", "full_steps"), ("space_recent32", "recent32_steps")):
        require(abs(weighted_mean(pairs, metric, weights) - original["means"][arm]["capped_time"]) < 1e-10,
                "primary mean agrees with original report")
    positive = sorted((p for p in pairs if p["gain_steps"] > 0),
                      key=lambda p: (-p["weighted_contribution"], p["seed"]))
    positive_gross = math.fsum(p["weighted_contribution"] for p in positive)
    negative_gross = math.fsum(p["weighted_contribution"] for p in pairs if p["gain_steps"] < 0)
    require(gain > 0 and positive_gross > 0, "original favorable mean retained")
    concentration = []
    for k in (1, 3, 5):
        top = positive[:k]
        contribution = math.fsum(p["weighted_contribution"] for p in top)
        concentration.append({"k": k, "seeds": [p["seed"] for p in top],
                              "contribution_steps": contribution,
                              "fraction_of_positive_gross": contribution / positive_gross,
                              "fraction_of_net_gain": contribution / gain})
    leave_pair = [{"removed_seed": p["seed"], **omission_estimate(
        [other for other in pairs if other["seed"] != p["seed"]], weights)} for p in pairs]
    leave_block = [{"removed_block": b, **omission_estimate(
        [p for p in pairs if p["block"] != b], weights)} for b in range(8)]
    blocks = []
    for block in range(8):
        rows = [p for p in pairs if p["block"] == block]
        blocks.append({"block": block, "weighted_mean_gain": weighted_mean(rows, "gain_steps", weights),
                       **counts(rows)})
    return {"scope": SCOPE, "original_continuation": {
                "learned_pilot_opportunity": False, "conditions_passed": 9, "conditions_total": 10,
                "criteria": original["criteria"]},
            "definitions": {"gain": "recent32 steps minus full-history steps; positive favors full history",
                "contribution": "original initial-hit mixture weight times paired gain divided by32",
                "block_index": "zero-based original fixed block; labels0..7",
                "concentration": "largest positive weighted contributions, ties ordered by seed",
                "gross_and_net": "positive gross excludes losses; fraction of net can exceed1 because losses subtract",
                "omission": "original mixture weights, recomputed within-stratum means; no new gate or uncertainty interval",
                "source_information": "hidden source locations and distances are neither used nor emitted"},
            "initial_hit_weights": weights, "pairs": pairs, "counts": counts(pairs),
            "weighted_mean_gain": gain, "positive_gross": positive_gross, "negative_gross": negative_gross,
            "strata": {h: counts([p for p in pairs if p["initial_hit"] == h]) for h in STRATA},
            "blocks": blocks, "concentration": concentration,
            "leave_one_pair_out": {"all_means_positive": all(r["mean_gain_steps"] > 0 for r in leave_pair),
                "minimum": min(leave_pair, key=lambda r: r["mean_gain_steps"]),
                "maximum": max(leave_pair, key=lambda r: r["mean_gain_steps"]), "estimates": leave_pair},
            "leave_one_block_out": {"all_means_positive": all(r["mean_gain_steps"] > 0 for r in leave_block),
                "minimum": min(leave_block, key=lambda r: r["mean_gain_steps"]),
                "maximum": max(leave_block, key=lambda r: r["mean_gain_steps"]), "estimates": leave_block}}


def self_check():
    # Unequal stratum counts: the raw case mean is2, the required mixture mean3.
    rows = [{"initial_hit": 1, "gain": 8}, {"initial_hit": 1, "gain": 0},
            {"initial_hit": 2, "gain": -4}, {"initial_hit": 3, "gain": 4}]
    weights = {1: .75, 2: .125, 3: .125}
    require(weighted_mean(rows, "gain", weights) == 3, "unequal-stratum hand-computed self-check")
    require(weighted_mean(rows[1:], "gain", weights) == 0, "omission preserves mixture weights")
    return {"status": "passed", "cases": 2, "runtime_or_environment_calls": 0}


def execute(args):
    start = time.perf_counter()
    source = Path(__file__).resolve()
    source_pin = sha(source)
    run = args.run.resolve()
    output = args.output.absolute()
    require(output.resolve() != run and run not in output.resolve().parents, "output outside frozen run")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "started", "scope": SCOPE, "source": str(source), "source_sha256": source_pin,
               "native_calls": 0, "model_calls": 0, "training_updates": 0, "output": str(output)}
    try:
        receipt["self_check"] = self_check()
        run, producer, original, weights, inputs = authenticate(args)
        receipt["inputs"] = inputs
        pairs = pairs_from_saved(run, producer, weights)
        result = analyze(pairs, weights, original)
        write(output / "analysis.json", result)
        # Detect input/source changes during reading before recording completion.
        _, _, _, _, current_inputs = authenticate(args)
        require(current_inputs == inputs and sha(source) == source_pin, "held inputs and reader source")
        receipt.update(status="completed", pairs=len(pairs), original_continuation_passed=False,
                       files={"analysis.json": {"sha256": sha(output / "analysis.json"),
                                                "bytes": (output / "analysis.json").stat().st_size}},
                       elapsed_seconds=time.perf_counter() - start)
        write(output / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - start)
        try:
            write(output / "failed.json", receipt)
        except BaseException as secondary:  # noqa: BLE001 - Preserve the primary analysis error.
            error.add_note(f"Failure receipt publication also failed: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    try:
        completed = execute(parser.parse_args())
        print(json.dumps({"status": completed["status"], "pairs": completed["pairs"],
                          "output": completed["output"]}), flush=True)
    except BaseException as error:
        print(f"Paired outcome analysis failed: {error!r}", file=sys.stderr, flush=True)
        raise
