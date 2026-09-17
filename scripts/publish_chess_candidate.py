"""Lossless candidate-study publication and standard-library portable audit.

Publish reproduces the frozen report without inference or engine calls. Portable
checks validate archived bytes, receipts and arithmetic, not chess legality or
checkpoint tensor contents. The full source report performs those checks first.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("direct", "action_only", "delta", "full_afterstate")
OPPONENTS = ("direct", "action_only", "full_afterstate")
SEEDS = (97, 109, 127)
SPLITS = ("dev", "shift")
ARCHIVE = "execution-and-report.tar.gz"
DATA_FILES = {
    "started.json",
    "excluded-states.json",
    "games.jsonl",
    "analyses.jsonl",
    "dev.jsonl",
    "shift.jsonl",
}
REGRET_FILES = {"engine.json", "analyses.jsonl", "regret.jsonl"}
FIT_FILES = {"weights.pt", "learning.jsonl", "training.json", "latency.json", "dev.jsonl", "shift.jsonl"}
COLORS = dict(zip(ARMS, ("#64748b", "#377eb8", "#168277", "#ae699a"), strict=True))
LABELS = dict(zip(ARMS, ("Direct", "Action only", "Exact delta", "Full afterstate"), strict=True))
ORIGINAL_PLAN_SHA256 = "8a49f3b7b16f8eb5b43fd7a54aca7c7510c31075988960bc44feb16b6bd84b3b"
RECOVERY_POLICY = "One separately frozen recovery attempt from fresh initialization. Same models, seeds, data, labels, updates, evaluation panels and gates. Discarded 128 updates count as extra study cost. No partial checkpoint, optimizer or RNG state reused. No further retries or replacement games."
RECOVERY_DATA_SCOPE = "Byte-identical previously generated development panels; no neural predictions existed at amendment freeze. Data generated once in failed attempt, copied and revalidated in recovery; not a second fresh sample."
RECOVERY_TIMING_SCOPE = "Failed execution wall time includes data generation, validation, cache construction and partial fit; separate partial-fit time unavailable. Do not add reused data generation time again."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def number(value, low=None, high=None):
    require(type(value) in (float, int) and math.isfinite(value), "Invalid finite numeric metric")
    require((low is None or value >= low) and (high is None or value <= high), "Metric outside bounds")
    return value


def close(actual, expected):
    number(actual)
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), "Inconsistent summary aggregate")


def mean(values):
    values = list(values)
    require(bool(values), "Cannot aggregate an empty panel")
    return math.fsum(values) / len(values)


def inventory(execution, report):
    result = {}
    for prefix, directory in (("execution", Path(execution)), ("report", Path(report))):
        require(directory.is_dir() and not directory.is_symlink(), "Evidence roots must be real directories")
        for path in sorted(directory.rglob("*")):
            require(
                not path.is_symlink() and (path.is_file() or path.is_dir()), "Nonregular evidence artifact"
            )
            if path.is_file():
                require(path.name != "failed.json", "Failed evidence cannot be published")
                result[f"{prefix}/{path.relative_to(directory).as_posix()}"] = {
                    "sha256": sha(path),
                    "size": path.stat().st_size,
                }
    return result


def make_archive(path, execution, report, members):
    roots = {"execution": Path(execution), "report": Path(report)}
    with (
        Path(path).open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, metadata in sorted(members.items()):
            prefix, relative = name.split("/", 1)
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = metadata["size"], 0o644, 0
            with (roots[prefix] / relative).open("rb") as stream:
                archive.addfile(info, stream)


def expected_names():
    return {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}


def arena_files(plan):
    return {"started.json", "summary.json"} | {
        f"{game['id']}.{ext}" for game in plan["arena_schedule"] for ext in ("json", "pgn")
    }


def fit_files(name):
    return FIT_FILES | (
        {"permuted-dev.jsonl", "permuted-shift.jsonl"} if name.startswith("delta-") else set()
    )


def digest_value(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def valid_hash(value):
    require(
        type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef"), "Invalid SHA256"
    )


def validate_recovery(recovery):
    require(
        set(recovery)
        == {
            "version",
            "original_plan",
            "original_plan_sha256",
            "failed_execution",
            "files",
            "failure",
            "completed_optimizer_updates",
            "discarded_training_examples",
            "discarded_computation",
            "completed_checkpoints",
            "neural_evaluation_rows",
            "reused_data_receipt_sha256",
            "reused_teacher_calls",
            "reused_requested_nodes",
            "policy",
            "data_scope",
            "timing_scope",
        },
        "Recovery schema differs",
    )
    require(
        recovery["version"] == "candidate-output-pipe-recovery-v1"
        and recovery["original_plan"] == "evidence/chess-candidate-v1/protocol/plan.json"
        and recovery["original_plan_sha256"] == ORIGINAL_PLAN_SHA256
        and recovery["failed_execution"] == "runs/chess-candidate-v1/execution"
        and recovery["completed_optimizer_updates"] == 128
        and recovery["discarded_training_examples"] == 16384
        and recovery["completed_checkpoints"] == 0
        and recovery["neural_evaluation_rows"] == 0
        and recovery["policy"] == RECOVERY_POLICY
        and recovery["data_scope"] == RECOVERY_DATA_SCOPE
        and recovery["timing_scope"] == RECOVERY_TIMING_SCOPE,
        "Recovery identity or exposure differs",
    )
    failure = recovery["failure"]
    require(
        set(failure) == {"status", "plan_sha256", "error", "wall_seconds"}
        and failure["status"] == "failed"
        and failure["plan_sha256"] == ORIGINAL_PLAN_SHA256
        and failure["error"] == "[Errno 32] Broken pipe",
        "Recovery failure identity differs",
    )
    number(failure["wall_seconds"], 0)
    members = recovery["files"]
    expected = {f"data/{n}" for n in DATA_FILES | {"completed.json"}} | {
        "started.json",
        "failed.json",
        "panels.json",
        "baselines.json",
        "training-cache.json",
        "action_only-97/learning.jsonl",
    }
    require(set(members) == expected, "Recovery failed-attempt membership differs")
    for item in members.values():
        require(set(item) == {"sha256", "bytes"}, "Recovery inventory schema differs")
        valid_hash(item["sha256"])
        require(type(item["bytes"]) is int and item["bytes"] > 0, "Invalid recovery artifact size")
    require(
        recovery["reused_data_receipt_sha256"] == members["data/completed.json"]["sha256"]
        and recovery["reused_teacher_calls"] == 4756
        and recovery["reused_requested_nodes"] == 9512000,
        "Recovery reused-data accounting differs",
    )
    count = recovery["discarded_computation"]["candidate_evaluations"]
    require(type(count) is int and count >= 16384, "Invalid discarded candidate count")
    require(
        recovery["discarded_computation"] == computation("action_only", 16384, count),
        "Discarded computation accounting differs",
    )


def recovery_disclosure(plan):
    if plan["protocol"]["version"] != "chess-candidate-v2":
        return ""
    return (
        "This is the separately frozen recovery of an output-pipe failure. All twelve models start from fresh "
        "initialization. The failed attempt's 128 optimizer updates were discarded and count as study overhead, "
        "not extra training of a published model. The same 4,096 previously generated evaluation positions are "
        "reused byte-for-byte, with no earlier neural predictions; they are not a second fresh sample. "
        "Teacher calls and data-generation time are counted once."
    )


def validate_plan(plan):
    p = plan["protocol"]
    require(
        p["version"] in ("chess-candidate-v1", "chess-candidate-v2"), "Unsupported candidate study version"
    )
    if p["version"] == "chess-candidate-v2":
        require("recovery" in plan, "Recovery plan is missing its failed-attempt record")
        validate_recovery(plan["recovery"])
    else:
        require("recovery" not in plan, "Original study cannot silently include recovery")
    require(
        p["arms"] == list(ARMS)
        and p["seeds"] == list(SEEDS)
        and p["width"] == 32
        and p["depth"] == 4
        and p["branch_depth"] == 2,
        "Unexpected candidate panel",
    )
    require(
        p["train_examples"] == 32768
        and p["epochs"] == 6
        and p["batch_size"] == 128
        and p["microbatch_size"] == 128
        and p["updates_per_fit"] == 1536,
        "Unexpected training budget",
    )
    require(
        p["relative_regret_reduction"] == 0.2
        and p["arena_score_threshold"] == 0.6
        and p["gate_absolute_tolerance"] == 1e-12,
        "Continuation thresholds differ",
    )
    require(
        len(plan["configurations"]) == 12 and {c["name"] for c in plan["configurations"]} == expected_names(),
        "Missing fits",
    )
    for c in plan["configurations"]:
        require(
            c == {"name": f"{c['arm']}-{c['seed']}", "arm": c["arm"], "seed": c["seed"], "width": 32}
            and c["arm"] in ARMS
            and c["seed"] in SEEDS,
            "Fit identity differs",
        )
    require(set(plan["parameters"]) == set(ARMS), "Parameter panel differs")
    for arm in ARMS:
        require(
            plan["parameters"][arm]
            == {
                "stored": 43854,
                "active": 33185 if arm == "direct" else 33313,
                "inactive_auxiliary": 10541,
                "inactive_action_projection": 128 if arm == "direct" else 0,
            },
            "Parameter counts differ",
        )
    require(
        {v["name"]: v["examples"] for v in plan["data_config"]["splits"]} == {"dev": 2048, "shift": 2048},
        "Data quotas differ",
    )
    openings, expected = plan["openings"], []
    require(len(openings) == len({o["id"] for o in openings}) == 16, "Incomplete openings")
    for opponent in OPPONENTS:
        for opening in openings:
            require(len(opening["moves"]) == 6, "Opening prefix differs")
            for seed in SEEDS:
                for white, black in (("delta", opponent), (opponent, "delta")):
                    expected.append(
                        {
                            "id": f"game-{len(expected) + 1:03d}",
                            "opening_id": opening["id"],
                            "opening_name": opening["name"],
                            "opening_moves": opening["moves"],
                            "seed": seed,
                            "opponent": opponent,
                            "white": f"{white}-{seed}",
                            "black": f"{black}-{seed}",
                            "white_arm": white,
                            "black_arm": black,
                        }
                    )
    require(plan["arena_schedule"] == expected, "Incomplete or reordered 288-game arena")
    rules = plan["arena_protocol"]
    require(
        rules["games"] == expected
        and rules["openings"] == openings
        and rules["depth"] == 4
        and rules["branch_depth"] == 2
        and rules["width"] == 32
        and rules["clock_seconds"] == 300
        and rules["max_played_plies"] == 240
        and rules["claim_draw"] is False
        and rules["increment"] == 0
        and rules["torch_threads"] == 2
        and rules["device"] == "cpu"
        and rules["bootstrap_replicates"] == 2000
        and rules["bootstrap_seed"] == 11300071
        and rules["primary_opponents"] == ["direct", "action_only"]
        and rules["gate_lower_score"] == 0.6,
        "Arena protocol mismatch",
    )


def arena_points(games, specs):
    statuses, outcomes = Counter(), Counter()
    for game, spec in zip(games, specs, strict=True):
        require(
            game["game_id"] == spec["id"]
            and all(game[k] == spec[k] for k in ("opening_id", "seed", "white", "black", "opponent")),
            "Game identity mismatch",
        )
        status, result = game["status"], game["result"]
        require(status in ("completed", "unfinished", "failed"), "Unknown status")
        require(type(game["played_plies"]) is int and 0 <= game["played_plies"] <= 240, "Invalid game length")
        statuses[status] += 1
        if status == "completed":
            require(result in ("1-0", "0-1", "1/2-1/2"), "Missing completed result")
            kind = (
                "draws"
                if result == "1/2-1/2"
                else "wins"
                if result == ("1-0" if spec["white_arm"] == "delta" else "0-1")
                else "losses"
            )
            outcomes[kind] += 1
        else:
            require(result == "*", "Unresolved game assigned a score")
    points = outcomes["wins"] + outcomes["draws"] / 2
    return (
        {k: statuses[k] for k in ("completed", "unfinished", "failed")},
        {
            **{k: outcomes[k] for k in ("wins", "draws", "losses")},
            "completed_points": points,
            "possible_points": len(games),
            "score_lower_bound": points / len(games),
            "score_upper_bound": (points + statuses["unfinished"] + statuses["failed"]) / len(games),
        },
    )


def validate_arena(plan, arena):
    require(
        arena["status"] == "completed"
        and arena["games"] == 288
        and arena["elo_estimate"] is None
        and arena["scope"] == plan["arena_protocol"]["scope"],
        "Arena scope differs",
    )
    games, specs = arena["game_results"], plan["arena_schedule"]
    require(len(games) == 288 and set(arena["by_opponent"]) == set(OPPONENTS), "Missing arena panel")
    for opponent, panel in [(None, arena["aggregate"]), *arena["by_opponent"].items()]:
        selected = [
            (g, sp)
            for g, sp in zip(games, specs, strict=True)
            if opponent is None or sp["opponent"] == opponent
        ]
        statuses, points = arena_points([g for g, _ in selected], [sp for _, sp in selected])
        require(
            panel["games"] == len(selected)
            and panel["status_counts"] == statuses
            and panel["delta"] == points,
            "Arena score bounds differ",
        )
        interval = panel["opening_cluster_bootstrap"]
        require(
            interval["openings"] == 16 and interval["replicates"] == 2000 and interval["seed"] == 11300071,
            "Arena interval identity differs",
        )
        for key in ("lower_bound_interval", "upper_bound_interval"):
            require(
                len(interval[key]) == 2 and 0 <= number(interval[key][0]) <= number(interval[key][1]) <= 1,
                "Invalid interval",
            )
        require(
            panel["played_plies"] == sum(g["played_plies"] for g, _ in selected), "Arena move total differs"
        )
        number(panel["policy_wall_seconds"], 0)
        require(
            type(panel["policy_calls"]) is int and panel["policy_calls"] >= panel["played_plies"],
            "Invalid policy calls",
        )
        if opponent is not None:
            required = opponent in ("direct", "action_only")
            gate = {
                "required": required,
                "threshold": 0.6 if required else None,
                "lower_score_bound": points["score_lower_bound"],
                "zero_failed_games": statuses["failed"] == 0,
                "passed": points["score_lower_bound"] >= 0.6 and statuses["failed"] == 0
                if required
                else None,
            }
            require(panel["gate"] == gate, "Per-opponent gate differs")
    require(arena["status_counts"] == arena["aggregate"]["status_counts"], "Aggregate statuses differ")
    checks = [{"opponent": o, **arena["by_opponent"][o]["gate"]} for o in ("direct", "action_only")]
    zero = arena["status_counts"]["failed"] == 0
    gate = {
        "primary_opponents": ["direct", "action_only"],
        "zero_failed_games": zero,
        "checks": checks,
        "passed": zero and all(c["passed"] for c in checks),
    }
    require(arena["gate"] == gate and arena["gate_passed"] is gate["passed"], "Arena continuation differs")
    for key in ("played_plies", "policy_calls", "policy_wall_seconds"):
        close(arena[key], arena["aggregate"][key])
        close(arena[key], math.fsum(arena["by_opponent"][o][key] for o in OPPONENTS))


def computation(arm, examples, candidates):
    root = examples * 4
    refinement = 0 if arm == "direct" else candidates * 2
    successor = candidates * 4 if arm == "full_afterstate" else 0
    return {
        "candidate_evaluations": candidates,
        "root_core_iterations": root,
        "candidate_refinement_iterations": refinement,
        "successor_root_iterations": successor,
        "total_core_iterations": root + refinement + successor,
        "delta_convolutions": candidates if arm == "delta" else 0,
        "successor_encoders": candidates if arm == "full_afterstate" else 0,
    }


def validate_metrics(row):
    require(
        row["examples"] == 2048 and type(row["correct"]) is int and 0 <= row["correct"] <= 2048,
        "Evaluation count differs",
    )
    close(row["agreement"], row["correct"] / 2048)
    for key in ("target_nll", "entropy", "hidden_rms", "logit_span"):
        number(row[key], 0)
    number(row["value_mae"], 0, 2)
    number(row["mean_confidence"], 0, 1)
    if row["correct"] == 2048:
        require(row["mismatch_confidence"] is None, "Unexpected mismatch confidence")
    else:
        number(row["mismatch_confidence"], 0, 1)


def validate_summary(plan, summary):
    validate_plan(plan)
    p = plan["protocol"]
    require(
        summary["status"] == "completed"
        and summary["novelty_established"] is False
        and summary["elo_estimate"] is None
        and summary["scope"] == p["scope"],
        "Summary scope differs",
    )
    for field in ("metrics", "costs", "latency", "regret"):
        require(set(summary[field]) == expected_names(), "Incomplete fit panel")
    require(
        set(summary["permutation_diagnostic"]) == {f"delta-{s}" for s in SEEDS},
        "Permutation fit coverage differs",
    )
    for name in expected_names():
        require(
            set(summary["metrics"][name]) == set(SPLITS) and set(summary["regret"][name]) == set(SPLITS),
            "Incomplete split coverage",
        )
        arm = name.rsplit("-", 1)[0]
        for split in SPLITS:
            validate_metrics(summary["metrics"][name][split])
            number(summary["regret"][name][split], -2, 2)
        if arm == "delta":
            require(
                set(summary["permutation_diagnostic"][name]) == set(SPLITS), "Incomplete diagnostic panels"
            )
            for split in SPLITS:
                value = summary["permutation_diagnostic"][name][split]
                validate_metrics(value["metrics"])
                require(
                    value["diagnostic"]["kind"] == "permuted_successors"
                    and value["diagnostic"]["positions"] == 2048
                    and value["diagnostic"]["permutation_seed"] == p["permutation_seed"],
                    "Diagnostic identity differs",
                )
        cost = summary["costs"][name]
        require(cost["updates"] == 1536 and cost["examples_seen"] == 196608, "Training budget differs")
        number(cost["training_seconds"], 0)
        require(
            cost["computation"]
            == computation(arm, 196608, summary["training_cache"]["legal_candidates"] * 6),
            "Training core/candidate totals differ",
        )
        timing = summary["latency"][name]
        require(
            timing["device"] == "cpu" and timing["depth"] == 4 and timing["torch_threads"] == 2,
            "Timing scope differs",
        )
        for key, count, total in (("records", 128, "total_wall_ms"), ("warmup_records", 3, "warmup_wall_ms")):
            require(len(timing[key]) == count, "Timing coverage differs")
            close(timing[total], math.fsum(number(r["wall_ms"], 0) for r in timing[key]))
    require(set(summary["means"]) == set(ARMS), "Incomplete arm means")
    for arm in ARMS:
        require(set(summary["means"][arm]) == set(SPLITS), "Incomplete mean splits")
        for split in SPLITS:
            for key in ("agreement", "target_nll", "value_mae", "mean_confidence"):
                close(
                    summary["means"][arm][split][key],
                    mean(summary["metrics"][f"{arm}-{z}"][split][key] for z in SEEDS),
                )
    validate_arena(plan, summary["arena"])
    require(len(summary["comparisons"]) == 6, "Comparison coverage differs")
    checks = []
    for (arm, split), row in zip(
        ((a, sp) for a in OPPONENTS for sp in SPLITS), summary["comparisons"], strict=True
    ):
        ref = mean(summary["regret"][f"{arm}-{z}"][split] for z in SEEDS)
        delta = mean(summary["regret"][f"delta-{z}"][split] for z in SEEDS)
        reduction = (ref - delta) / ref if ref > 0 else None
        require(
            row["comparator"] == arm
            and row["split"] == split
            and row["primary"] is (arm != "full_afterstate"),
            "Comparison identity differs",
        )
        for key, value in (
            ("reference_bounded_regret", ref),
            ("delta_bounded_regret", delta),
            ("bounded_regret_change", delta - ref),
        ):
            close(row[key], value)
        if reduction is None:
            require(row["relative_reduction"] is None, "Undefined reduction must stay null")
        else:
            close(row["relative_reduction"], reduction)
        interval = row["agreement_interval"]
        close(
            interval["mean"],
            summary["means"]["delta"][split]["agreement"] - summary["means"][arm][split]["agreement"],
        )
        require(
            interval["positions"] == 2048
            and 0 < interval["games"] <= 2048
            and interval["scope"] == p["uncertainty"]
            and -1 <= number(interval["lower"]) <= number(interval["upper"]) <= 1,
            "Interval scope differs",
        )
        if arm != "full_afterstate":
            passed = reduction is not None and (
                reduction >= 0.2 or math.isclose(reduction, 0.2, rel_tol=0, abs_tol=1e-12)
            )
            checks.append(
                {
                    "metric": "relative_regret_reduction",
                    "split": split,
                    "comparator": arm,
                    "observed": reduction,
                    "threshold": 0.2,
                    "passed": passed,
                }
            )
    checks.append({"metric": "arena", "passed": summary["arena"]["gate_passed"]})
    require(len(summary["checks"]) == 5, "Gate coverage differs")
    for actual, expected in zip(summary["checks"], checks, strict=True):
        if expected.get("observed") is not None:
            close(actual["observed"], expected["observed"])
        else:
            require(actual.get("observed") is None, "Undefined gate reduction differs")
        require(
            {k: v for k, v in actual.items() if k != "observed"}
            == {k: v for k, v in expected.items() if k != "observed"},
            "Gate arithmetic differs",
        )
    require(
        summary["continuation_passed"] is all(c["passed"] for c in checks), "Continuation verdict differs"
    )
    if p["version"] == "chess-candidate-v2":
        require(summary.get("recovery") == plan["recovery"], "Summary recovery lineage differs")
        account = summary.get("attempt_accounting", {})
        require(
            set(account)
            == {
                "failed_wall_seconds",
                "successful_execution_wall_seconds",
                "total_attempt_wall_seconds",
                "final_fit_updates",
                "discarded_updates",
                "total_optimizer_updates",
                "data_generated_once",
            }
            and account["data_generated_once"] is True
            and account["final_fit_updates"] == 18432
            and account["discarded_updates"] == 128
            and account["total_optimizer_updates"] == 18560,
            "Recovery attempt accounting differs",
        )
        close(account["failed_wall_seconds"], plan["recovery"]["failure"]["wall_seconds"])
        number(account["successful_execution_wall_seconds"], 0)
        close(
            account["total_attempt_wall_seconds"],
            account["failed_wall_seconds"] + account["successful_execution_wall_seconds"],
        )
    else:
        require(
            "recovery" not in summary and "attempt_accounting" not in summary,
            "Original summary cannot silently include recovery",
        )
    return summary


def required_members(plan):
    result = {
        "execution/started.json",
        "execution/completed.json",
        "execution/baselines.json",
        "execution/panels.json",
        "execution/training-cache.json",
        "execution/evaluation-caches.json",
        "report/summary.json",
        "report/completed.json",
    }
    result |= {f"execution/data/{name}" for name in DATA_FILES | {"completed.json"}}
    result |= {f"execution/regret/{name}" for name in REGRET_FILES | {"completed.json"}}
    result |= {f"execution/arena/{name}" for name in arena_files(plan) | {"completed.json"}}
    result |= {
        f"execution/{fit}/{name}" for fit in expected_names() for name in fit_files(fit) | {"completed.json"}
    }
    return result


def keep_document(name):
    if name.endswith(".json"):
        return name != "execution/data/excluded-states.json"
    return name in {
        "execution/data/dev.jsonl",
        "execution/data/shift.jsonl",
        "execution/regret/regret.jsonl",
        "execution/regret/analyses.jsonl",
    } or any(
        name == f"execution/{fit}/{suffix}"
        for fit in expected_names()
        for suffix in (
            "dev.jsonl",
            "shift.jsonl",
            "learning.jsonl",
            "permuted-dev.jsonl",
            "permuted-shift.jsonl",
        )
    )


def document(raw, name):
    if name.endswith(".jsonl"):
        require(raw.endswith(b"\n"), "Incomplete JSONL record")
        return [decode(line) for line in raw.splitlines()]
    result = decode(raw)
    if name.startswith("execution/arena/game-"):
        result = dict(result)
        result["attempt_count"] = len(result.pop("attempts"))
        result["move_count"] = len(result.pop("moves"))
    return result


def inspect_archive(path, expected):
    actual, documents = {}, {}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name, parts = member.name, PurePosixPath(member.name).parts
            require(
                member.isfile()
                and name not in actual
                and name in expected
                and parts
                and parts[0] in ("execution", "report")
                and ".." not in parts
                and str(PurePosixPath(name)) == name
                and "failed.json" not in parts,
                "Unexpected, duplicated or unsafe archive member",
            )
            require(member.size == expected[name]["size"], "Archive member size mismatch")
            stream, digest, chunks = archive.extractfile(member), hashlib.sha256(), []
            keep = keep_document(name)
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                if keep:
                    chunks.append(chunk)
            actual[name] = {"sha256": digest.hexdigest(), "size": member.size}
            require(actual[name] == expected[name], "Archive member hash mismatch")
            if keep:
                documents[name] = document(b"".join(chunks), name)
    require(actual == expected, "Archive membership differs from manifest")
    return actual, documents


def prediction_metrics(predictions, rows):
    require(len(predictions) == len(rows) == 2048, "Incomplete prediction panel")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate evaluation IDs")
    for pred, row in zip(predictions, rows, strict=True):
        require(
            pred["id"] == row["id"]
            and pred["game_id"] == row["game_id"]
            and type(pred["game_id"]) is type(row["game_id"])
            and pred["target"] == row["target_uci"]
            and pred["target_value"] == row["target_value"],
            "Prediction identity or target differs",
        )
        require(
            type(pred["correct"]) is bool and pred["correct"] == (pred["choice"] == pred["target"]),
            "Prediction correctness differs",
        )
        require(type(pred["choice"]) is str and bool(pred["choice"]), "Missing move choice")
        for key in ("target_nll", "entropy", "hidden_rms", "logit_span"):
            number(pred[key], 0)
        number(pred["value"], -1, 1)
        number(pred["target_value"], -1, 1)
        number(pred["target_probability"], 0, 1)
        number(pred["max_probability"], 0, 1)
        require(pred["max_probability"] > 0, "Zero maximum probability")
        require(
            math.isclose(
                pred["target_probability"], math.exp(-pred["target_nll"]), rel_tol=1e-10, abs_tol=1e-14
            ),
            "Probability/NLL mismatch",
        )
        require(
            pred["target_probability"] <= pred["max_probability"] + 1e-12
            and pred["entropy"] + 1e-10 >= -math.log(pred["max_probability"]),
            "Probability bound mismatch",
        )
        if pred["correct"]:
            require(
                math.isclose(
                    pred["target_probability"], pred["max_probability"], rel_tol=1e-10, abs_tol=1e-14
                ),
                "Correct choice is not the maximum",
            )
    wrong = [r["max_probability"] for r in predictions if not r["correct"]]
    correct = sum(r["correct"] for r in predictions)
    return {
        "examples": len(rows),
        "correct": correct,
        "agreement": correct / len(rows),
        "target_nll": mean(r["target_nll"] for r in predictions),
        "value_mae": mean(abs(r["value"] - r["target_value"]) for r in predictions),
        "mean_confidence": mean(r["max_probability"] for r in predictions),
        "mismatch_confidence": mean(wrong) if wrong else None,
        **{k: mean(r[k] for r in predictions) for k in ("entropy", "hidden_rms", "logit_span")},
    }


def validate_cache(cache, plan, positions, rows=None):
    identity_keys = (
        "version",
        "encoding",
        "input_rows_sha256",
        "labels_sha256",
        "include_successors",
        "tensors",
    )
    require(
        cache["version"] == "candidate-position-cache-v1"
        and cache["include_successors"] is True
        and cache["positions"] == positions
        and cache["root_encodings"] == positions,
        "Cache identity differs",
    )
    identity = {k: cache[k] for k in identity_keys}
    require(cache["cache_sha256"] == digest_value(identity), "Cache fingerprint differs")
    for key in ("input_rows_sha256", "labels_sha256", "source_sha256", "encoding_source_sha256"):
        valid_hash(cache[key])
    require(
        cache["source_sha256"] == plan["sources"]["src/openjev/research/chess_candidate_eval.py"]
        and cache["encoding_source_sha256"] == plan["sources"]["src/openjev/research/chess_spatial.py"],
        "Cache source differs",
    )
    if rows is not None:
        require(
            cache["input_rows_sha256"]
            == digest_value([{k: r[k] for k in ("id", "game_id", "fen")} for r in rows])
            and cache["labels_sha256"]
            == digest_value([{k: r[k] for k in ("id", "target_uci", "target_value")} for r in rows]),
            "Cache inputs/labels differ",
        )
    count = cache["legal_candidates"]
    require(type(count) is int and count >= positions, "Invalid cached legal count")
    require(
        all(
            cache[k] == count
            for k in ("native_successors", "native_board_copies", "native_pushes", "successor_encodings")
        ),
        "Native construction count differs",
    )
    specs = {
        "observations": ([positions, 19, 8, 8], "torch.float32", 4),
        "candidates": ([count, 5], "torch.int64", 8),
        "counts": ([positions], "torch.int64", 8),
        "offsets": ([positions + 1], "torch.int64", 8),
        "targets": ([positions], "torch.int64", 8),
        "values": ([positions], "torch.float32", 4),
        "successors": ([count, 19, 8, 8], "torch.float32", 4),
    }
    require(set(cache["tensors"]) == set(specs), "Cache tensor coverage differs")
    for key, (shape, dtype, size) in specs.items():
        t = cache["tensors"][key]
        require(
            t["shape"] == shape and t["dtype"] == dtype and t["bytes"] == math.prod(shape) * size,
            "Cache storage differs",
        )
        valid_hash(t["sha256"])
    require(
        cache["cpu_tensor_bytes"] == sum(t["bytes"] for t in cache["tensors"].values()),
        "Cache byte total differs",
    )
    for key in ("native_construction_wall_seconds", "fingerprinting_wall_seconds", "total_wall_seconds"):
        number(cache[key], 0)
    require(
        cache["total_wall_seconds"]
        >= cache["native_construction_wall_seconds"] + cache["fingerprinting_wall_seconds"],
        "Cache timing differs",
    )


def diagnostic_metrics(preds, arm, permuted):
    kind = "permuted_successors" if permuted else "intact"
    for row in preds:
        require(row["arm"] == arm and row["evaluation_kind"] == kind, "Prediction intervention differs")
        off, singleton = row["successor_permutation_offset"], row["one_legal_move_unchanged"]
        require(type(off) is int and off >= 0 and type(singleton) is bool, "Invalid permutation metadata")
        require(
            (off == 0 and not singleton) if not permuted else ((off == 0) == singleton),
            "Permutation/singleton identity differs",
        )
    return {
        "kind": kind,
        "permutation_seed": 11300083 if permuted else None,
        "positions": len(preds),
        "permuted_positions": sum(r["successor_permutation_offset"] > 0 for r in preds),
        "one_legal_move_unchanged": sum(r["one_legal_move_unchanged"] for r in preds),
        "mapping_sha256": digest_value([[r["id"], r["successor_permutation_offset"]] for r in preds]),
    }


def validate_evaluation(receipt, preds, arm, permuted, cache_hash):
    require(
        receipt["diagnostic"] == diagnostic_metrics(preds, arm, permuted)
        and receipt["cache_sha256"] == cache_hash
        and receipt["timing_scope"]
        == "Cached CPU batch evaluation and prediction writing; native cache construction excluded and recorded separately.",
        "Evaluation cache or diagnostic receipt differs",
    )
    number(receipt["evaluation_wall_seconds"], 0)


def validate_chain(plan, plan_hash, summary, members, documents):
    require(required_members(plan) <= set(members), "Archive lacks required full-panel evidence")
    require(summary["plan_sha256"] == plan_hash, "Summary plan mismatch")

    def digest(name):
        return members[name]["sha256"]

    def files(prefix, receipt, required):
        require(
            receipt["status"] == "completed" and set(receipt["files"]) == required,
            "Receipt file coverage mismatch",
        )
        require(
            all(digest(f"{prefix}/{n}") == h for n, h in receipt["files"].items()),
            "Archived receipt hash mismatch",
        )

    complete, report = documents["execution/completed.json"], documents["report/completed.json"]
    started = documents["execution/started.json"]
    require(
        complete["status"] == "completed"
        and complete["plan_sha256"] == plan_hash
        and started["status"] == "started"
        and started["plan_sha256"] == plan_hash,
        "Execution completion identity mismatch",
    )
    number(complete["wall_seconds"], 0)
    files(
        "execution",
        complete,
        {
            n.removeprefix("execution/")
            for n in members
            if n.startswith("execution/") and n != "execution/completed.json"
        },
    )
    require(
        report["status"] == "completed"
        and report["plan_sha256"] == plan_hash
        and report["summary_sha256"] == digest("report/summary.json")
        and report["execution_receipt_sha256"] == digest("execution/completed.json")
        and summary["execution_receipt_sha256"] == digest("execution/completed.json")
        and documents["report/summary.json"] == summary,
        "Report chain mismatch",
    )
    data, regret = documents["execution/data/completed.json"], documents["execution/regret/completed.json"]
    files("execution/data", data, DATA_FILES)
    files("execution/regret", regret, REGRET_FILES)
    require(
        data["counts"] == {"dev": 2048, "shift": 2048}
        and documents["execution/data/started.json"]["config"] == plan["data_config"],
        "Data quotas/config differ",
    )
    require(
        summary["fresh_data_cost"] == data and summary["engine_cost"] == regret["cost"],
        "Summary cost receipt mismatch",
    )
    if plan["protocol"]["version"] == "chess-candidate-v2":
        recovery = plan["recovery"]
        close(summary["attempt_accounting"]["successful_execution_wall_seconds"], complete["wall_seconds"])
        require(
            data["teacher_calls"] == recovery["reused_teacher_calls"]
            and data["requested_nodes"] == recovery["reused_requested_nodes"],
            "Reused teacher cost differs",
        )
        for name in DATA_FILES | {"completed.json"}:
            original = recovery["files"][f"data/{name}"]
            current = members[f"execution/data/{name}"]
            require(
                current == {"sha256": original["sha256"], "size": original["bytes"]},
                "Reused evaluation data differs from the failed attempt",
            )
    cost, p = regret["cost"], plan["protocol"]
    require(
        type(cost["calls"]) is int
        and 0 < cost["calls"] <= p["regret_call_ceiling"]
        and cost["requested_nodes"] == cost["calls"] * p["regret_nodes"]
        and cost["requested_nodes"] <= p["regret_node_ceiling"],
        "Engine call budget mismatch",
    )
    engine = documents["execution/regret/engine.json"]
    require(
        engine["sha256"] == plan["engine_sha256"] and engine["id"]["name"].startswith("Stockfish 19"),
        "Engine identity mismatch",
    )
    panel = documents["execution/panels.json"]
    indices = panel["latency_indices"]
    require(
        len(indices) == 128
        and len(set(indices)) == 128
        and all(type(i) is int and 0 <= i < 4096 for i in indices)
        and sum(i < 2048 for i in indices) == 64,
        "Latency panel differs",
    )
    require(
        panel["graded_configurations"] == [{"id": c["name"]} for c in plan["configurations"]],
        "Graded fit identities differ",
    )
    eval_rows = {s: documents[f"execution/data/{s}.jsonl"] for s in SPLITS}
    for split in SPLITS:
        selected = panel["secondary_indices"][split]
        require(
            len(selected) == 128
            and selected == sorted(set(selected))
            and all(type(i) is int and 0 <= i < 2048 for i in selected),
            "Grading panel differs",
        )
    combined = eval_rows["dev"] + eval_rows["shift"]
    predictions, shared_orders, shared_initial = {}, {}, {}
    require(
        summary["training_cache"] == documents["execution/training-cache.json"]
        and summary["evaluation_caches"] == documents["execution/evaluation-caches.json"],
        "Cache summary differs",
    )
    validate_cache(summary["training_cache"], plan, 32768)
    require(set(summary["evaluation_caches"]) == set(SPLITS), "Evaluation cache coverage differs")
    for split in SPLITS:
        validate_cache(summary["evaluation_caches"][split], plan, 2048, eval_rows[split])
    for config in plan["configurations"]:
        name, prefix = config["name"], f"execution/{config['name']}"
        receipt, trained = documents[prefix + "/completed.json"], documents[prefix + "/training.json"]
        files(prefix, receipt, fit_files(name))
        require(
            set(receipt["diagnostic"]) == (set(SPLITS) if config["arm"] == "delta" else set()),
            "Diagnostic receipt membership differs",
        )
        require(
            trained["cache_sha256"] == summary["training_cache"]["cache_sha256"], "Training cache differs"
        )
        for identity in (receipt, trained):
            require(
                identity["status"] == "completed"
                and identity["plan_sha256"] == plan_hash
                and all(identity[k] == v for k, v in config.items()),
                "Fit identity mismatch",
            )
        require(
            trained["weights_sha256"] == digest(prefix + "/weights.pt")
            and trained["learning_sha256"] == digest(prefix + "/learning.jsonl")
            and trained["data_receipt_sha256"] == digest("execution/data/completed.json"),
            "Training input binding mismatch",
        )
        require(
            {k: trained[k] for k in summary["costs"][name]} == summary["costs"][name], "Fit cost mismatch"
        )
        state = trained["initial_state_sha256"]
        require(
            type(state) is str and len(state) == 64 and set(state) <= set("0123456789abcdef"),
            "Invalid initialization digest",
        )
        require(
            shared_initial.setdefault(config["seed"], state) == state, "Shared model initialization differs"
        )
        learning = documents[prefix + "/learning.jsonl"]
        order = [r["indices_sha256"] for r in learning]
        require(shared_orders.setdefault(config["seed"], order) == order, "Paired minibatch ordering differs")
        totals = computation(config["arm"], 0, 0)
        require(len(learning) == 1536, "Incomplete optimizer steps")
        for i, row in enumerate(learning):
            require(
                row["step"] == i + 1
                and row["epoch"] == i // 256 + 1
                and row["root_depth"] == 4
                and row["branch_depth"] == 2
                and row["examples"] == 128,
                "Learning journal schedule differs",
            )
            for field in ("policy_ce", "value_mse", "loss", "gradient_norm"):
                number(row[field], 0)
            valid_hash(row["indices_sha256"])
            require(
                row["microbatches"] == 1
                and type(row["sampled_mps_allocated_bytes_after_backward"]) is int
                and row["sampled_mps_allocated_bytes_after_backward"] >= 0,
                "Microbatch memory receipt differs",
            )
            candidates = row["candidate_evaluations"]
            require(type(candidates) is int and candidates >= 128, "Candidate count differs")
            for key, value in computation(config["arm"], 128, candidates).items():
                require(row[key] == value, "Per-step candidate computation differs")
                totals[key] += value
            require(
                math.isclose(
                    row["loss"], row["policy_ce"] + 0.5 * row["value_mse"], rel_tol=2e-6, abs_tol=1e-6
                ),
                "Learning loss arithmetic differs",
            )
        require(totals == trained["computation"], "Logged computation sum differs")
        require(set(receipt["evaluation"]) == set(SPLITS), "Fit split coverage differs")
        predictions[name] = {}
        for split in SPLITS:
            preds = documents[f"{prefix}/{split}.jsonl"]
            predictions[name][split] = preds
            metrics = prediction_metrics(preds, eval_rows[split])
            require(
                metrics == summary["metrics"][name][split] == receipt["evaluation"][split]["metrics"],
                "Recorded predictions do not reproduce metrics",
            )
            validate_evaluation(
                receipt["evaluation"][split],
                preds,
                config["arm"],
                False,
                summary["evaluation_caches"][split]["cache_sha256"],
            )
            if config["arm"] == "delta":
                altered = documents[f"{prefix}/permuted-{split}.jsonl"]
                calculated = prediction_metrics(altered, eval_rows[split])
                diagnostic = diagnostic_metrics(altered, "delta", True)
                require(
                    summary["permutation_diagnostic"][name][split]
                    == {"metrics": calculated, "diagnostic": diagnostic}
                    and receipt["diagnostic"][split]["metrics"] == calculated,
                    "Permutation metrics differ",
                )
                validate_evaluation(
                    receipt["diagnostic"][split],
                    altered,
                    "delta",
                    True,
                    summary["evaluation_caches"][split]["cache_sha256"],
                )
        timing = documents[prefix + "/latency.json"]
        require(timing == summary["latency"][name], "Latency summary differs")
        require(
            [r["index"] for r in timing["records"]] == indices
            and all(r["id"] == combined[i]["id"] for r, i in zip(timing["records"], indices, strict=True)),
            "Latency position identity mismatch",
        )
        require(
            [r["index"] for r in timing["warmup_records"]] == [0, 1, 2]
            and all(r["id"] == "starting-board" for r in timing["warmup_records"]),
            "Warmup identity differs",
        )
    baseline = documents["execution/baselines.json"]
    require(
        summary["baselines"] == {s: {n: v["metrics"] for n, v in baseline[s].items()} for s in SPLITS},
        "Baseline summary differs",
    )
    for row in summary["comparisons"]:
        require(
            row["agreement_interval"]["games"] == len({r["game_id"] for r in eval_rows[row["split"]]}),
            "Interval game-cluster count differs",
        )
    analyses = documents["execution/regret/analyses.jsonl"]
    require(len(analyses) == cost["calls"], "Engine call count differs")
    close(cost["wall_seconds"], math.fsum(number(a["wall_seconds"], 0) for a in analyses))
    require(cost["reported_nodes"] == sum(a["reported_nodes"] for a in analyses), "Engine node sum differs")
    lookup = {}
    for row in analyses:
        split, i = row["split"], row["panel_index"]
        require(
            split in SPLITS
            and i in panel["secondary_indices"][split]
            and row["id"] == eval_rows[split][i]["id"]
            and row["requested_nodes"] == 20000,
            "Engine analysis identity differs",
        )
        number(row["bounded_score"], -1, 1)
        key = (split, i, row["root_move"])
        require(key not in lookup, "Duplicate engine analysis")
        lookup[key] = row
    records = documents["execution/regret/regret.jsonl"]
    require(len(records) == 3072, "Incomplete engine grading records")
    expected, seen, required_calls = {}, set(), set()
    for name in expected_names():
        for split in SPLITS:
            expected[(name, split)] = []
    for row in records:
        name, split, i = row["configuration"], row["split"], row["panel_index"]
        require(
            (name, split) in expected and i in panel["secondary_indices"][split], "Grading selection differs"
        )
        key = name, split, i
        require(key not in seen, "Repeated graded decision")
        seen.add(key)
        pred = predictions[name][split][i]
        require(row["id"] == pred["id"] and row["choice"] == pred["choice"], "Graded decision differs")
        ref, chosen = (split, i, None), (split, i, row["choice"])
        require(ref in lookup and chosen in lookup, "Missing paired engine analysis")
        close(row["bounded_regret"], lookup[ref]["bounded_score"] - lookup[chosen]["bounded_score"])
        close(row["cp_loss"], lookup[ref]["score_cp"] - lookup[chosen]["score_cp"])
        required_calls.update((ref, chosen))
        expected[(name, split)].append(row["bounded_regret"])
    require(required_calls == set(lookup), "Unexpected engine calls")
    for (name, split), values in expected.items():
        require(len(values) == 128, "Incomplete per-fit grading coverage")
        close(summary["regret"][name][split], mean(values))
    arena_prefix = "execution/arena"
    arena_receipt, arena_start = (
        documents[arena_prefix + "/completed.json"],
        documents[arena_prefix + "/started.json"],
    )
    files(arena_prefix, arena_receipt, arena_files(plan))
    require(
        arena_receipt["plan_sha256"] == plan_hash
        and arena_start["plan_sha256"] == plan_hash
        and arena_start["status"] == "started"
        and arena_start["protocol"] == plan["arena_protocol"]
        and set(arena_start["models"]) == expected_names(),
        "Arena source binding differs",
    )
    for c in plan["configurations"]:
        model = arena_start["models"][c["name"]]
        require(
            model["width"] == c["width"]
            and model["seed"] == c["seed"]
            and model["depth"] == 4
            and model["arm"] == c["arm"]
            and model["branch_depth"] == 2
            and model["recurrence"] == "residual"
            and len(model["state_sha256"]) == 64
            and set(model["state_sha256"]) <= set("0123456789abcdef"),
            "Arena model identity differs",
        )
    require(documents[arena_prefix + "/summary.json"] == summary["arena"], "Arena summary differs")
    games = [documents[f"{arena_prefix}/{spec['id']}.json"] for spec in plan["arena_schedule"]]
    for game, result, spec in zip(
        games, summary["arena"]["game_results"], plan["arena_schedule"], strict=True
    ):
        require(
            all(game[k] == v for k, v in result.items())
            and game["opening_moves"] == spec["opening_moves"]
            and game["white_arm"] == spec["white_arm"]
            and game["black_arm"] == spec["black_arm"]
            and game["opponent"] == spec["opponent"]
            and game["opening_plies"] == 6
            and game["move_count"] == game["played_plies"] + 6,
            "Raw game and summary identity differ",
        )
    require(
        sum(g["attempt_count"] for g in games) == summary["arena"]["policy_calls"], "Arena call sum differs"
    )
    close(summary["arena"]["policy_wall_seconds"], math.fsum(number(g["wall_seconds"], 0) for g in games))


def load_study(plan=None):
    version = "chess-candidate-v1" if plan is None else plan["protocol"]["version"]
    sources = {
        "chess-candidate-v1": "scripts/chess_candidate_study.py",
        "chess-candidate-v2": "scripts/chess_candidate_recovery.py",
    }
    require(version in sources, "Unsupported candidate study version")
    spec = importlib.util.spec_from_file_location("_candidate_publication_study", ROOT / sources[version])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_card(plan, plan_hash, weights):
    lines = [
        "# OpenJev candidate chess checkpoints",
        "",
        "License: MIT. All twelve final fits are retained.",
        "",
        "Four width-32 controls share root initialization, 32,768 training positions, six epochs and 1,536 updates. Root depth is four; refinement depth is two. Direct has 33,185 active parameters, each refinement arm 33,313; all store 43,854 including unused heads. Parameters, FLOPs and wall time are not matched.",
        "",
        "Delta and full-afterstate policies receive native one-ply consequences in the root player's perspective. These are not learned dynamics, multi-ply search, RL or memory between moves. No Elo or novelty claim follows. All scores are uncalibrated candidate-relative softmax values.",
        "",
        f"Frozen plan SHA-256: `{plan_hash}`.",
        "",
        "| Checkpoint | Arm | Seed |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| [{name}]({row['path']}) | {row['arm']} | {row['seed']} |" for name, row in sorted(weights.items())
    )
    if recovery_disclosure(plan):
        lines += [
            "",
            recovery_disclosure(plan),
            "",
            "[Preserved failed attempt](../../evidence/chess-candidate-v1/failed-attempt/README.md).",
        ]
    lines += [
        "",
        "Load with `CandidateChess.load(path, expected_plan_sha256=..., expected_arm=..., expected_seed=..., expected_width=32, expected_root_depth=4, expected_branch_depth=2)`. All original bytes are copied; no winner alias or selected seed.",
        "",
        "Source hashes:",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(plan["sources"].items()))
    return "\n".join(lines) + "\n"


def publish(plan_path, execution, report, out, models=None):
    plan_path, execution, report, out = map(Path, (plan_path, execution, report, out))
    supplied_plan = read(plan_path)
    validate_plan(supplied_plan)
    version = supplied_plan["protocol"]["version"]
    models = ROOT / "models" / version if models is None else Path(models)
    if out.exists() or models.exists():
        raise FileExistsError("Publication or models directory already exists")
    for destination in (out.resolve(), models.resolve()):
        require(
            not any(
                destination.is_relative_to(root.resolve()) or root.resolve().is_relative_to(destination)
                for root in (execution, report)
            ),
            "Publication destinations must be outside source evidence",
        )
    require(
        not out.resolve().is_relative_to(models.resolve())
        and not models.resolve().is_relative_to(out.resolve()),
        "Publication and model directories must not overlap",
    )
    study = load_study(supplied_plan)
    plan = study.verify_plan(plan_path)
    summary = validate_summary(plan, read(report / "summary.json"))
    members = inventory(execution, report)
    documents = {
        name: document(
            ((execution if name.startswith("execution/") else report) / name.split("/", 1)[1]).read_bytes(),
            name,
        )
        for name in members
        if keep_document(name)
    }
    validate_chain(plan, sha(plan_path), summary, members, documents)
    with tempfile.TemporaryDirectory(prefix="openjev-candidate-publication-") as temporary:
        reproduced = study.report(plan_path, execution, Path(temporary) / "report")
    require(summary == reproduced, "Supplied summary does not reproduce from the frozen report")
    out.mkdir(parents=True, exist_ok=False)
    models.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, out / "plan.json")
    shutil.copyfile(report / "summary.json", out / "summary.json")
    make_archive(out / ARCHIVE, execution, report, members)
    require(members == inventory(execution, report), "Evidence changed while being packaged")
    weights = {}
    for config in sorted(plan["configurations"], key=lambda c: c["name"]):
        name = config["name"]
        original, target = f"execution/{name}/weights.pt", models / name / "weights.pt"
        target.parent.mkdir()
        shutil.copyfile(execution / name / "weights.pt", target)
        require(
            sha(target) == members[original]["sha256"] and target.stat().st_size == members[original]["size"],
            "Copied checkpoint differs from original",
        )
        weights[name] = {
            **config,
            **members[original],
            "path": f"{name}/weights.pt",
            "original_member": original,
            "plan_sha256": sha(plan_path),
        }
    (models / "README.md").write_text(model_card(plan, sha(plan_path), weights))
    shutil.copyfile(ROOT / "LICENSE", models / "LICENSE")
    models_relative = Path(os.path.relpath(models.resolve(), out.resolve())).as_posix()
    verdict = lambda value: "PASSED" if value else "FAILED"
    recovery_text = (
        "\n"
        + recovery_disclosure(plan)
        + " [Preserved failed attempt](../../chess-candidate-v1/failed-attempt/README.md).\n"
        if recovery_disclosure(plan)
        else ""
    )
    (out / "README.md").write_text(
        "# Candidate-conditioned chess development evidence\n\n"
        f"Engine-score gate: **{verdict(all(c['passed'] for c in summary['checks'][:4]))}**. "
        f"Game gate: **{verdict(summary['arena']['gate_passed'])}**. "
        f"Combined continuation: **{verdict(summary['continuation_passed'])}**.\n\n"
        f"[Summary](summary.json) · [Plan](plan.json) · [All raw evidence]({ARCHIVE}) · "
        f"[All twelve checkpoints]({models_relative}/README.md) · [Hash manifest](manifest.json)\n\n"
        "The archive preserves every execution and report file, including evaluation positions, generator traces, "
        "raw engine calls, training logs, predictions, original weights and all 288 game JSON/PGN traces. "
        "Derived cache metadata and fingerprints are retained; the temporary float feature buffers are not archived. "
        "Unfinished or failed games retain unresolved points; they are never relabeled draws. "
        "This is a development mechanism comparison on known generators, conditional on three training seeds. Native one-ply consequences are not learned dynamics. Mapping corruption remains diagnostic only. "
        "No Elo estimate, new architecture or novelty claim is established.\n" + recovery_text
    )
    manifest = {
        "version": 1,
        "plan_sha256": sha(out / "plan.json"),
        "archive": {"path": ARCHIVE, "sha256": sha(out / ARCHIVE), "size": (out / ARCHIVE).stat().st_size},
        "members": members,
        "member_count": len(members),
        "raw_bytes": sum(r["size"] for r in members.values()),
        "models_directory": models_relative,
        "weights": weights,
        "model_files": {name: sha(models / name) for name in ("README.md", "LICENSE")},
        "publication_readme_sha256": sha(out / "README.md"),
    }
    write_new(out / "manifest.json", manifest)
    actual, documents = inspect_archive(out / ARCHIVE, members)
    validate_chain(plan, sha(out / "plan.json"), summary, actual, documents)
    write_new(
        out / "completed.json",
        {
            "status": "completed",
            "fits": 12,
            "games": 288,
            "plan_sha256": sha(out / "plan.json"),
            "summary_sha256": sha(out / "summary.json"),
            "manifest_sha256": sha(out / "manifest.json"),
            "publisher_sha256": sha(__file__),
            "source_report_reproduced": True,
            "archive_roundtrip_verified": True,
            "copied_weights_match_originals": True,
        },
    )
    return audit(out)


def audit(publication):
    out = Path(publication)
    receipt, manifest = read(out / "completed.json"), read(out / "manifest.json")
    plan, summary = read(out / "plan.json"), read(out / "summary.json")
    validate_summary(plan, summary)
    plan_hash = sha(out / "plan.json")
    require(
        receipt["status"] == "completed"
        and receipt["fits"] == 12
        and receipt["games"] == 288
        and receipt["plan_sha256"] == plan_hash == manifest["plan_sha256"]
        and receipt["summary_sha256"] == sha(out / "summary.json")
        and receipt["manifest_sha256"] == sha(out / "manifest.json")
        and not (out / "failed.json").exists(),
        "Publication completion receipt mismatch",
    )
    require(
        all(
            receipt.get(k) is True
            for k in (
                "source_report_reproduced",
                "archive_roundtrip_verified",
                "copied_weights_match_originals",
            )
        ),
        "Missing verification receipt",
    )
    require(
        receipt.get("publisher_sha256") == sha(__file__),
        "Audit implementation differs from the recorded publisher",
    )
    archive = manifest["archive"]
    require(
        archive["path"] == ARCHIVE
        and archive["sha256"] == sha(out / ARCHIVE)
        and archive["size"] == (out / ARCHIVE).stat().st_size,
        "Compressed archive checksum mismatch",
    )
    members, documents = inspect_archive(out / ARCHIVE, manifest["members"])
    require(
        manifest["member_count"] == len(members)
        and manifest["raw_bytes"] == sum(r["size"] for r in members.values())
        and members["report/summary.json"]["sha256"] == sha(out / "summary.json")
        and set(manifest["weights"]) == expected_names(),
        "Manifest totals or checkpoint coverage mismatch",
    )
    validate_chain(plan, plan_hash, summary, members, documents)
    require(
        not Path(manifest["models_directory"]).is_absolute(),
        "Model directory must be a portable relative path",
    )
    models = out / manifest["models_directory"]
    require(models.is_dir() and not models.is_symlink(), "Model directory must be a real directory")
    for config in plan["configurations"]:
        name = config["name"]
        row = manifest["weights"][name]
        original, relative = f"execution/{name}/weights.pt", f"{name}/weights.pt"
        path = models / relative
        require(
            all(row[k] == v for k, v in config.items())
            and row["original_member"] == original
            and row["path"] == relative
            and row["plan_sha256"] == plan_hash
            and row["sha256"] == members[original]["sha256"]
            and row["size"] == members[original]["size"]
            and not path.is_symlink()
            and not path.parent.is_symlink()
            and sha(path) == row["sha256"]
            and path.stat().st_size == row["size"],
            "Published checkpoint identity or original bytes mismatch",
        )
    require(set(manifest["model_files"]) == {"README.md", "LICENSE"}, "Missing model card or MIT license")
    for name, digest in manifest["model_files"].items():
        require(
            not (models / name).is_symlink() and sha(models / name) == digest,
            "Model documentation checksum mismatch",
        )
    require(sha(out / "README.md") == manifest["publication_readme_sha256"], "Publication index changed")
    return {
        "status": "verified",
        "fits": 12,
        "games": 288,
        "evaluations": 24,
        "permutation_panels": 6,
        "members": len(members),
        "plan_sha256": plan_hash,
        "archive_sha256": archive["sha256"],
        "summary_sha256": sha(out / "summary.json"),
        "scope": "Portable byte, receipt and scalar arithmetic audit. Chess legality, tensor identities and "
        "bootstrap resampling are checked by frozen report reproduction at publication time.",
    }


def figure(plan, summary):
    validate_summary(plan, summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, grid = plt.subplots(2, 3, figsize=(15, 10))
    axes = grid.ravel()
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.80, wspace=0.32, hspace=0.53)
    fig.suptitle(
        "OpenJev: do exact candidate consequences help?",
        x=0.065,
        ha="left",
        y=0.975,
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.934,
        "12 fits · 32,768 shared training positions · 288 paired development games",
        fontsize=11,
        color="#475569",
    )
    fig.legend(
        handles=[Line2D([], [], marker="o", ls="", color=COLORS[a], label=LABELS[a]) for a in ARMS]
        + [
            Line2D([], [], marker=m, ls="", color="#475569", label=f"Seed {z}")
            for m, z in zip(("o", "s", "^"), SEEDS, strict=True)
        ],
        loc="upper left",
        bbox_to_anchor=(0.06, 0.91),
        ncol=7,
        frameon=False,
    )
    markers = ("o", "s", "^")
    if summary["scope"].lower().startswith("synthetic"):
        fig.text(0.065, 0.845, "SYNTHETIC QA DATA - not study results", color="#b91c1c", fontsize=11)
    for ax, key, title, ylabel, factor in (
        (axes[0], "agreement", "A   Teacher agreement", "Agreement (%)", 100),
        (axes[1], "regret", "B   Stronger-engine assessment", "Signed bounded loss (lower is better)", 1),
    ):
        for j, split in enumerate(SPLITS):
            for k, arm in enumerate(ARMS):
                x = j + (k - 1.5) * 0.18
                values = [
                    summary["regret"][f"{arm}-{z}"][split]
                    if key == "regret"
                    else summary["metrics"][f"{arm}-{z}"][split][key]
                    for z in SEEDS
                ]
                ax.bar(x, mean(values) * factor, width=0.15, color=COLORS[arm], alpha=0.25)
                for i, value in enumerate(values):
                    ax.scatter(
                        x + (i - 1) * 0.032,
                        value * factor,
                        color=COLORS[arm],
                        marker=markers[i],
                        s=28,
                        zorder=3,
                    )
        ax.set(xticks=(0, 1), xticklabels=("Ordinary", "Shifted"), title=title, ylabel=ylabel)
        ax.axhline(0, color="#cbd5e1", lw=0.8)
    ax = axes[2]
    for arm in ARMS:
        for i, seed in enumerate(SEEDS):
            name = f"{arm}-{seed}"
            ms = summary["latency"][name]["total_wall_ms"] / 128
            quality = mean(summary["metrics"][name][s]["agreement"] for s in SPLITS) * 100
            ax.scatter(ms, quality, color=COLORS[arm], marker=markers[i], s=40)
    ax.set(
        title="C   Full CPU decision cost",
        xlabel="Mean milliseconds per decision",
        ylabel="Mean agreement across panels (%)",
    )
    ax = axes[3]
    for i, opponent in enumerate(OPPONENTS):
        points = summary["arena"]["by_opponent"][opponent]["delta"]
        lo, hi = points["score_lower_bound"] * 100, points["score_upper_bound"] * 100
        ax.hlines(i, lo, hi, color=COLORS[opponent], lw=5)
        ax.scatter([lo, hi], [i, i], color=COLORS[opponent], marker="|")
        ax.annotate(
            f"{points['wins']}W / {points['draws']}D / {points['losses']}L",
            (lo, i),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(60, color="#b91c1c", ls="--", lw=1)
    ax.set(
        title="D   Delta points versus each opponent",
        xlabel="Lower–upper points (%) over all 96 games",
        yticks=range(3),
        yticklabels=[LABELS[o] for o in OPPONENTS],
        xlim=(-2, 102),
        ylim=(-0.7, 2.7),
    )
    ax = axes[4]
    for j, split in enumerate(SPLITS):
        for i, seed in enumerate(SEEDS):
            intact = summary["metrics"][f"delta-{seed}"][split]["agreement"] * 100
            altered = summary["permutation_diagnostic"][f"delta-{seed}"][split]["metrics"]["agreement"] * 100
            ax.plot(
                [j * 3, j * 3 + 1],
                [intact, altered],
                marker=markers[i],
                ms=4,
                color=COLORS["delta"],
                alpha=0.65,
            )
    ax.set(
        title="E   Delta mapping corruption diagnostic",
        ylabel="Agreement (%)",
        xticks=(0, 1, 3, 4),
        xticklabels=("Intact\nordinary", "Permuted\nordinary", "Intact\nshifted", "Permuted\nshifted"),
    )
    ax = axes[5]
    for j, arm in enumerate(ARMS):
        seconds = [summary["costs"][f"{arm}-{z}"]["training_seconds"] for z in SEEDS]
        ax.bar(j, mean(seconds), color=COLORS[arm], alpha=0.25)
        for i, value in enumerate(seconds):
            ax.scatter(j + (i - 1) * 0.08, value, color=COLORS[arm], marker=markers[i], s=30)
    ax.set(
        title="F   Recorded fit cost",
        ylabel="Training seconds per fit",
        xticks=range(4),
        xticklabels=("Direct", "Action\nonly", "Exact\ndelta", "Full\nafterstate"),
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    verdict = "PASSED" if summary["continuation_passed"] else "FAILED"
    fig.text(
        0.065,
        0.096,
        f"Frozen continuation: {verdict}. Dots show all three fitted seeds. Unfinished/failed games remain unresolved.",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.065,
        "Native one-ply consequences, not learned dynamics. Parameters, FLOPs and wall time are unequal. No Elo or novelty claim.",
        fontsize=9,
        color="#475569",
    )
    fig.text(
        0.065,
        0.038,
        "Mapping corruption is diagnostic only; it cannot replace the primary engine-loss and arena criteria. CPU calls include native transitions and encoding.",
        fontsize=9,
        color="#475569",
    )
    if recovery_disclosure(plan):
        fig.text(
            0.065,
            0.012,
            "Recovery: same evaluation data reused; all fits restarted. 128 discarded updates are extra study cost, excluded from fit-cost panel F.",
            fontsize=9,
            color="#475569",
        )
    return fig


def render(publication, prefix):
    publication, prefix = Path(publication), Path(prefix)
    verification = audit(publication)
    paths = {ext: prefix.with_suffix("." + ext) for ext in ("png", "svg", "json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Figure output already exists")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(read(publication / "plan.json"), read(publication / "summary.json"))
    fig.savefig(paths["png"], dpi=180)
    fig.savefig(paths["svg"], metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    write_new(
        paths["json"],
        {
            "status": "completed",
            "verification": verification,
            "publisher_sha256": sha(__file__),
            "plots": {ext: sha(paths[ext]) for ext in ("png", "svg")},
        },
    )
    return {ext: str(path) for ext, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("publish")
    for name in ("plan", "execution", "report", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--models", type=Path, default=ROOT / "models/chess-candidate-v1")
    for command in ("audit", "render"):
        p = sub.add_parser(command)
        p.add_argument("--publication", type=Path, required=True)
        if command == "render":
            p.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "publish":
        result = publish(args.plan, args.execution, args.report, args.out, args.models)
    elif args.command == "audit":
        result = audit(args.publication)
    else:
        result = render(args.publication, args.prefix)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
