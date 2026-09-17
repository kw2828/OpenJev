"""Synthetic publication fixtures only; no model, data generation or engine calls."""

import copy
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjev.research import chess_candidate_arena as arena

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


p = module("_candidate_publisher_test", ROOT / "scripts/publish_chess_candidate.py")
study = module("_candidate_fixture_protocol", ROOT / "scripts/chess_candidate_study.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def plan_fixture():
    protocol = copy.deepcopy(study.PROTOCOL)
    protocol["scope"] = "Synthetic publication fixture only; not experimental results."
    return {
        "protocol": protocol,
        "configurations": study.configs(),
        "parameters": {
            a: {
                "stored": 43854,
                "active": 33185 if a == "direct" else 33313,
                "inactive_auxiliary": 10541,
                "inactive_action_projection": 128 if a == "direct" else 0,
            }
            for a in p.ARMS
        },
        "data_config": {"splits": [{"name": s, "examples": 2048} for s in p.SPLITS]},
        "openings": json.loads(json.dumps(arena.OPENINGS)),
        "arena_schedule": arena.schedule(),
        "arena_protocol": arena.protocol(),
        "engine_sha256": "c" * 64,
        "sources": {
            "src/openjev/research/chess_candidate_eval.py": "d" * 64,
            "src/openjev/research/chess_spatial.py": "e" * 64,
        },
    }


def cache_fixture(n, plan, rows=None):
    total = n * 20
    specs = {
        "observations": ([n, 19, 8, 8], "torch.float32", 4),
        "candidates": ([total, 5], "torch.int64", 8),
        "counts": ([n], "torch.int64", 8),
        "offsets": ([n + 1], "torch.int64", 8),
        "targets": ([n], "torch.int64", 8),
        "values": ([n], "torch.float32", 4),
        "successors": ([total, 19, 8, 8], "torch.float32", 4),
    }
    tensors = {
        k: {"shape": shape, "dtype": dtype, "bytes": math.prod(shape) * size, "sha256": "f" * 64}
        for k, (shape, dtype, size) in specs.items()
    }
    identity = {
        "version": "candidate-position-cache-v1",
        "encoding": "side-relative-rank-mirror-19-v1",
        "include_successors": True,
        "tensors": tensors,
        "input_rows_sha256": p.digest_value([{k: r[k] for k in ("id", "game_id", "fen")} for r in rows])
        if rows
        else "a" * 64,
        "labels_sha256": p.digest_value(
            [{k: r[k] for k in ("id", "target_uci", "target_value")} for r in rows]
        )
        if rows
        else "b" * 64,
    }
    return {
        **identity,
        "cache_sha256": p.digest_value(identity),
        "positions": n,
        "legal_candidates": total,
        "root_encodings": n,
        **dict.fromkeys(
            ("native_successors", "native_board_copies", "native_pushes", "successor_encodings"), total
        ),
        "cpu_tensor_bytes": sum(t["bytes"] for t in tensors.values()),
        "native_construction_wall_seconds": 1.0,
        "fingerprinting_wall_seconds": 0.1,
        "total_wall_seconds": 1.2,
        "source_sha256": plan["sources"]["src/openjev/research/chess_candidate_eval.py"],
        "encoding_source_sha256": plan["sources"]["src/openjev/research/chess_spatial.py"],
    }


def make_games(plan):
    games = []
    for i, spec in enumerate(plan["arena_schedule"]):
        kind = ("win", "win", "draw", "loss", "unfinished", "failed")[i % 6]
        unresolved = kind in ("unfinished", "failed")
        result = (
            "*"
            if unresolved
            else "1/2-1/2"
            if kind == "draw"
            else ("1-0" if (kind == "win") == (spec["white_arm"] == "delta") else "0-1")
        )
        games.append(
            {
                **{k: v for k, v in spec.items() if k != "id"},
                "game_id": spec["id"],
                "status": kind if unresolved else "completed",
                "result": result,
                "termination": "Synthetic fixture",
                "played_plies": 2,
                "opening_plies": 6,
                "wall_seconds": 0.02,
                "attempts": [{}, {}],
                "moves": [{}] * 8,
            }
        )
    return games, arena.summarize(games)


def compare(summary, plan):
    summary["means"] = {
        a: {
            s: {
                k: p.mean(summary["metrics"][f"{a}-{z}"][s][k] for z in p.SEEDS)
                for k in ("agreement", "target_nll", "value_mae", "mean_confidence")
            }
            for s in p.SPLITS
        }
        for a in p.ARMS
    }
    summary["comparisons"], summary["checks"] = [], []
    for arm in p.OPPONENTS:
        for split in p.SPLITS:
            ref, delta = [
                p.mean(summary["regret"][f"{a}-{z}"][split] for z in p.SEEDS) for a in (arm, "delta")
            ]
            rel = (ref - delta) / ref if ref > 0 else None
            summary["comparisons"].append(
                {
                    "split": split,
                    "comparator": arm,
                    "primary": arm != "full_afterstate",
                    "reference_bounded_regret": ref,
                    "delta_bounded_regret": delta,
                    "bounded_regret_change": delta - ref,
                    "relative_reduction": rel,
                    "agreement_interval": {
                        "games": 32,
                        "positions": 2048,
                        "mean": summary["means"]["delta"][split]["agreement"]
                        - summary["means"][arm][split]["agreement"],
                        "lower": -0.5,
                        "upper": 0.5,
                        "scope": plan["protocol"]["uncertainty"],
                    },
                }
            )
            if arm != "full_afterstate":
                summary["checks"].append(
                    {
                        "metric": "relative_regret_reduction",
                        "split": split,
                        "comparator": arm,
                        "observed": rel,
                        "threshold": 0.2,
                        "passed": rel is not None
                        and (rel >= 0.2 or math.isclose(rel, 0.2, rel_tol=0, abs_tol=1e-12)),
                    }
                )
    summary["checks"].append({"metric": "arena", "passed": summary["arena"]["gate_passed"]})
    summary["continuation_passed"] = all(c["passed"] for c in summary["checks"])


def seal(execution, report, plan_path, summary):
    write(
        execution / "completed.json",
        {
            "status": "completed",
            "plan_sha256": p.sha(plan_path),
            "wall_seconds": 60.0,
            "files": {
                path.relative_to(execution).as_posix(): p.sha(path)
                for path in execution.rglob("*")
                if path.is_file() and path != execution / "completed.json"
            },
        },
    )
    summary["execution_receipt_sha256"] = p.sha(execution / "completed.json")
    write(report / "summary.json", summary)
    write(
        report / "completed.json",
        {
            "status": "completed",
            "plan_sha256": p.sha(plan_path),
            "summary_sha256": p.sha(report / "summary.json"),
            "execution_receipt_sha256": summary["execution_receipt_sha256"],
        },
    )


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidate-synthetic-evidence")
    execution, report, plan_path = root / "execution", root / "report", root / "plan.json"
    plan = plan_fixture()
    write(plan_path, plan)
    plan_hash = p.sha(plan_path)
    write(execution / "started.json", {"status": "started", "plan_sha256": plan_hash})
    panel = {
        "secondary_indices": {s: list(range(1800, 1928)) for s in p.SPLITS},
        "latency_indices": list(range(64)) + list(range(2048, 2112)),
        "graded_configurations": [{"id": c["name"]} for c in plan["configurations"]],
    }
    write(execution / "panels.json", panel)
    data = {
        s: [
            {
                "id": f"{s}-{i}",
                "game_id": i // 64,
                "fen": "synthetic FEN marker",
                "target_uci": "e2e4",
                "target_value": 0.0,
            }
            for i in range(2048)
        ]
        for s in p.SPLITS
    }
    for name in p.DATA_FILES:
        write(execution / "data" / name, {"synthetic": True})
    write(execution / "data/started.json", {"config": plan["data_config"]})
    for split, values in data.items():
        jsonl(execution / "data" / f"{split}.jsonl", values)
    dc = {
        "status": "completed",
        "counts": {"dev": 2048, "shift": 2048},
        "teacher_calls": 4756,
        "requested_nodes": 9512000,
        "files": {n: p.sha(execution / "data" / n) for n in p.DATA_FILES},
    }
    write(execution / "data/completed.json", dc)
    training_cache = cache_fixture(32768, plan)
    caches = {s: cache_fixture(2048, plan, data[s]) for s in p.SPLITS}
    write(execution / "training-cache.json", training_cache)
    write(execution / "evaluation-caches.json", caches)
    baseline = {
        s: {"greedy_material": {"metrics": {"examples": 2048, "top1_teacher_agreement": 0.125}}}
        for s in p.SPLITS
    }
    write(execution / "baselines.json", baseline)
    summary = {
        "status": "completed",
        "plan_sha256": plan_hash,
        "novelty_established": False,
        "elo_estimate": None,
        "scope": plan["protocol"]["scope"],
        "metrics": {},
        "costs": {},
        "latency": {},
        "regret": {},
        "permutation_diagnostic": {},
        "training_cache": training_cache,
        "evaluation_caches": caches,
        "fresh_data_cost": dc,
        "baselines": {s: {n: v["metrics"] for n, v in b.items()} for s, b in baseline.items()},
    }
    for c in plan["configurations"]:
        name, arm = c["name"], c["arm"]
        directory = execution / name
        summary["metrics"][name] = {}
        evaluation, diagnostic = {}, {}
        for split in p.SPLITS:
            for permuted in [False, True] if arm == "delta" else [False]:
                preds = []
                for i, row in enumerate(data[split]):
                    correct = i < (512 if arm == "direct" or permuted else 1024)
                    preds.append(
                        {
                            "id": row["id"],
                            "game_id": row["game_id"],
                            "target": "e2e4",
                            "target_value": 0.0,
                            "choice": "e2e4" if correct else "g1f3" if arm == "delta" else "d2d4",
                            "correct": correct,
                            "value": 0.1,
                            "target_nll": -math.log(0.8 if correct else 0.2),
                            "target_probability": 0.8 if correct else 0.2,
                            "max_probability": 0.8,
                            "entropy": 0.5,
                            "hidden_rms": 0.6,
                            "logit_span": 2.0,
                            "arm": arm,
                            "evaluation_kind": "permuted_successors" if permuted else "intact",
                            "successor_permutation_offset": 1 if permuted else 0,
                            "one_legal_move_unchanged": False,
                        }
                    )
                jsonl(directory / f"{'permuted-' if permuted else ''}{split}.jsonl", preds)
                metrics = p.prediction_metrics(preds, data[split])
                diag = p.diagnostic_metrics(preds, arm, permuted)
                rec = {
                    "metrics": metrics,
                    "diagnostic": diag,
                    "cache_sha256": caches[split]["cache_sha256"],
                    "evaluation_wall_seconds": 1.0,
                    "timing_scope": "Cached CPU batch evaluation and prediction writing; native cache construction excluded and recorded separately.",
                }
                if permuted:
                    diagnostic[split] = rec
                    summary["permutation_diagnostic"].setdefault(name, {})[split] = {
                        "metrics": metrics,
                        "diagnostic": diag,
                    }
                else:
                    evaluation[split] = rec
                    summary["metrics"][name][split] = metrics
        timing = {
            "device": "cpu",
            "torch_threads": 2,
            "depth": 4,
            "records": [
                {
                    "id": (data["dev"] + data["shift"])[i]["id"],
                    "index": i,
                    "choice": "e2e4",
                    "wall_ms": float(1 + p.ARMS.index(arm)),
                }
                for i in panel["latency_indices"]
            ],
            "warmup_records": [
                {"id": "starting-board", "index": i, "choice": "e2e4", "wall_ms": 1.0} for i in range(3)
            ],
            "total_wall_ms": 128.0 * (1 + p.ARMS.index(arm)),
            "warmup_wall_ms": 3.0,
        }
        write(directory / "latency.json", timing)
        learning = [
            {
                "step": i + 1,
                "epoch": i // 256 + 1,
                "root_depth": 4,
                "branch_depth": 2,
                "examples": 128,
                "indices_sha256": p.digest_value([c["seed"], i]),
                "microbatches": 1,
                "policy_ce": 1.0,
                "value_mse": 0.2,
                "loss": 1.1,
                "gradient_norm": 0.5,
                "sampled_mps_allocated_bytes_after_backward": 100,
                **p.computation(arm, 128, 2560),
            }
            for i in range(1536)
        ]
        jsonl(directory / "learning.jsonl", learning)
        (directory / "weights.pt").write_bytes(("Synthetic checkpoint " + name).encode())
        cost = {
            "updates": 1536,
            "examples_seen": 196608,
            "training_seconds": 10.0 * (1 + p.ARMS.index(arm)),
            "computation": p.computation(arm, 196608, 3932160),
            "memory_scope": "Synthetic memory receipt",
        }
        write(
            directory / "training.json",
            {
                "status": "completed",
                **c,
                **cost,
                "plan_sha256": plan_hash,
                "initial_state_sha256": "a" * 64,
                "cache_sha256": training_cache["cache_sha256"],
                "data_receipt_sha256": p.sha(execution / "data/completed.json"),
                "weights_sha256": p.sha(directory / "weights.pt"),
                "learning_sha256": p.sha(directory / "learning.jsonl"),
            },
        )
        write(
            directory / "completed.json",
            {
                "status": "completed",
                **c,
                "plan_sha256": plan_hash,
                "evaluation": evaluation,
                "diagnostic": diagnostic,
                "files": {n: p.sha(directory / n) for n in p.fit_files(name)},
            },
        )
        summary["costs"][name], summary["latency"][name] = cost, timing
        summary["regret"][name] = dict.fromkeys(p.SPLITS, 0.24 if arm == "delta" else 0.3)
    analyses, grades = [], []
    for split in p.SPLITS:
        for i in panel["secondary_indices"][split]:
            for move, score, cp in ((None, 0.6, 400), ("d2d4", 0.3, 200), ("g1f3", 0.36, 240)):
                analyses.append(
                    {
                        "split": split,
                        "panel_index": i,
                        "id": data[split][i]["id"],
                        "root_move": move,
                        "bounded_score": score,
                        "score_cp": cp,
                        "requested_nodes": 20000,
                        "reported_nodes": 20000,
                        "wall_seconds": 0.01,
                    }
                )
            for c in plan["configurations"]:
                delta = c["arm"] == "delta"
                grades.append(
                    {
                        "configuration": c["name"],
                        "split": split,
                        "panel_index": i,
                        "id": data[split][i]["id"],
                        "choice": "g1f3" if delta else "d2d4",
                        "bounded_regret": 0.24 if delta else 0.3,
                        "cp_loss": 160 if delta else 200,
                    }
                )
    jsonl(execution / "regret/analyses.jsonl", analyses)
    jsonl(execution / "regret/regret.jsonl", grades)
    write(
        execution / "regret/engine.json",
        {"sha256": plan["engine_sha256"], "id": {"name": "Stockfish 19 synthetic"}},
    )
    cost = {
        "calls": len(analyses),
        "requested_nodes": len(analyses) * 20000,
        "reported_nodes": len(analyses) * 20000,
        "wall_seconds": len(analyses) * 0.01,
    }
    write(
        execution / "regret/completed.json",
        {
            "status": "completed",
            "cost": cost,
            "files": {n: p.sha(execution / "regret" / n) for n in p.REGRET_FILES},
        },
    )
    summary["engine_cost"] = cost
    games, summary["arena"] = make_games(plan)
    write(
        execution / "arena/started.json",
        {
            "status": "started",
            "plan_sha256": plan_hash,
            "protocol": plan["arena_protocol"],
            "models": {
                c["name"]: {
                    "arm": c["arm"],
                    "width": 32,
                    "seed": c["seed"],
                    "depth": 4,
                    "branch_depth": 2,
                    "recurrence": "residual",
                    "state_sha256": "e" * 64,
                }
                for c in plan["configurations"]
            },
        },
    )
    for game in games:
        write(execution / "arena" / f"{game['game_id']}.json", game)
        (execution / "arena" / f"{game['game_id']}.pgn").write_text("Synthetic PGN marker\n")
    write(execution / "arena/summary.json", summary["arena"])
    write(
        execution / "arena/completed.json",
        {
            "status": "completed",
            "plan_sha256": plan_hash,
            "wall_seconds": 6.0,
            "files": {n: p.sha(execution / "arena" / n) for n in p.arena_files(plan)},
        },
    )
    compare(summary, plan)
    seal(execution, report, plan_path, summary)
    return {"plan": plan_path, "execution": execution, "report": report}


@pytest.fixture
def evidence(source, tmp_path, monkeypatch):
    root = tmp_path / "source"
    shutil.copytree(source["plan"].parent, root)
    result = {
        "plan": root / "plan.json",
        "execution": root / "execution",
        "report": root / "report",
        "out": tmp_path / "published",
        "models": tmp_path / "models",
    }
    monkeypatch.setattr(
        p,
        "load_study",
        lambda plan=None: SimpleNamespace(
            verify_plan=lambda path: p.read(path), report=lambda *_: p.read(result["report"] / "summary.json")
        ),
    )
    return result


def publish(evidence):
    return p.publish(*(evidence[k] for k in ("plan", "execution", "report", "out", "models")))


def recovery_fixture(execution):
    members = {
        name: {"sha256": "a" * 64, "bytes": 128}
        for name in (
            "started.json",
            "failed.json",
            "panels.json",
            "baselines.json",
            "training-cache.json",
            "action_only-97/learning.jsonl",
        )
    }
    for name in p.DATA_FILES | {"completed.json"}:
        path = execution / "data" / name
        members[f"data/{name}"] = {"sha256": p.sha(path), "bytes": path.stat().st_size}
    return {
        "version": "candidate-output-pipe-recovery-v1",
        "original_plan": "evidence/chess-candidate-v1/protocol/plan.json",
        "original_plan_sha256": p.ORIGINAL_PLAN_SHA256,
        "failed_execution": "runs/chess-candidate-v1/execution",
        "files": members,
        "failure": {
            "status": "failed",
            "plan_sha256": p.ORIGINAL_PLAN_SHA256,
            "error": "[Errno 32] Broken pipe",
            "wall_seconds": 457.0,
        },
        "completed_optimizer_updates": 128,
        "discarded_training_examples": 16384,
        "discarded_computation": p.computation("action_only", 16384, 327680),
        "completed_checkpoints": 0,
        "neural_evaluation_rows": 0,
        "reused_data_receipt_sha256": members["data/completed.json"]["sha256"],
        "reused_teacher_calls": 4756,
        "reused_requested_nodes": 9512000,
        "policy": p.RECOVERY_POLICY,
        "data_scope": p.RECOVERY_DATA_SCOPE,
        "timing_scope": p.RECOVERY_TIMING_SCOPE,
    }


def promote_recovery(evidence):
    """Synthetic v2 case with unchanged data bytes and rebound final-attempt receipts."""
    execution = evidence["execution"]
    old_hash = p.sha(evidence["plan"])
    plan = p.read(evidence["plan"])
    plan["protocol"]["version"] = "chess-candidate-v2"
    plan["recovery"] = recovery_fixture(execution)
    write(evidence["plan"], plan)
    new_hash = p.sha(evidence["plan"])

    def rebind(value):
        if isinstance(value, dict):
            return {k: rebind(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebind(v) for v in value]
        return new_hash if value == old_hash else value

    for path in execution.rglob("*.json"):
        if path.parent == execution / "data":
            continue
        write(path, rebind(p.read(path)))
    for name in p.expected_names():
        directory = execution / name
        receipt = p.read(directory / "completed.json")
        receipt["files"] = {n: p.sha(directory / n) for n in receipt["files"]}
        write(directory / "completed.json", receipt)
    path = execution / "arena/completed.json"
    receipt = p.read(path)
    receipt["files"] = {n: p.sha(path.parent / n) for n in receipt["files"]}
    write(path, receipt)
    summary = rebind(p.read(evidence["report"] / "summary.json"))
    summary["recovery"] = plan["recovery"]
    summary["attempt_accounting"] = {
        "failed_wall_seconds": 457.0,
        "successful_execution_wall_seconds": 60.0,
        "total_attempt_wall_seconds": 517.0,
        "final_fit_updates": 18432,
        "discarded_updates": 128,
        "total_optimizer_updates": 18560,
        "data_generated_once": True,
    }
    seal(execution, evidence["report"], evidence["plan"], summary)
    return plan, summary


def test_recovery_preserves_failed_cost_and_reused_data_in_portable_publication(evidence, tmp_path):
    plan, summary = promote_recovery(evidence)
    result = publish(evidence)
    assert result["status"] == "verified"
    assert p.read(evidence["out"] / "summary.json")["attempt_accounting"] == summary["attempt_accounting"]
    for path in (evidence["out"] / "README.md", evidence["models"] / "README.md"):
        text = path.read_text()
        assert "128 optimizer updates were discarded" in text
        assert "not a second fresh sample" in text
        assert "failed-attempt/README.md" in text
    outputs = p.render(evidence["out"], tmp_path / "recovery-figure")
    assert "128 discarded updates" in Path(outputs["svg"]).read_text()
    with tarfile.open(evidence["out"] / p.ARCHIVE) as archive:
        for name in p.DATA_FILES | {"completed.json"}:
            raw = archive.extractfile(f"execution/data/{name}").read()
            assert len(raw) == plan["recovery"]["files"][f"data/{name}"]["bytes"]
            assert p.hashlib.sha256(raw).hexdigest() == plan["recovery"]["files"][f"data/{name}"]["sha256"]


@pytest.mark.parametrize(
    "fault", ["version", "missing", "old_hash", "checkpoint", "neural_rows", "cost", "lineage", "data"]
)
def test_recovery_cannot_hide_exposure_cost_or_changed_data(evidence, fault):
    plan, summary = promote_recovery(evidence)
    if fault == "version":
        plan["protocol"]["version"] = "chess-candidate-v3"
    elif fault == "missing":
        del plan["recovery"]
    elif fault == "old_hash":
        plan["recovery"]["original_plan_sha256"] = "0" * 64
    elif fault == "checkpoint":
        plan["recovery"]["completed_checkpoints"] = 1
    elif fault == "neural_rows":
        plan["recovery"]["neural_evaluation_rows"] = 1
    elif fault == "cost":
        summary["attempt_accounting"]["total_optimizer_updates"] = 18432
    elif fault == "lineage":
        summary["recovery"] = copy.deepcopy(summary["recovery"])
        summary["recovery"]["discarded_training_examples"] = 0
    else:
        # Keep the supplied plan/summary mutually consistent while claiming different original bytes.
        plan["recovery"]["files"]["data/dev.jsonl"]["sha256"] = "0" * 64
        members = p.inventory(evidence["execution"], evidence["report"])
        documents = {
            name: p.document((evidence[prefix] / relative).read_bytes(), name)
            for name in members
            if p.keep_document(name)
            for prefix, relative in [name.split("/", 1)]
        }
        documents["report/summary.json"] = summary
        with pytest.raises(ValueError, match="Reused evaluation data"):
            p.validate_chain(plan, p.sha(evidence["plan"]), summary, members, documents)
        return
    with pytest.raises(ValueError):
        p.validate_summary(plan, summary)


def test_study_loader_rejects_unknown_version_and_retains_default():
    assert p.load_study().PROTOCOL["version"] == "chess-candidate-v1"
    assert (
        p.load_study({"protocol": {"version": "chess-candidate-v2"}}).PROTOCOL["version"]
        == "chess-candidate-v2"
    )
    with pytest.raises(ValueError, match="Unsupported"):
        p.load_study({"protocol": {"version": "chess-candidate-v9"}})


def test_complete_archive_models_and_python_without_site_packages(evidence):
    result = publish(evidence)
    assert (result["fits"], result["games"], result["evaluations"], result["permutation_panels"]) == (
        12,
        288,
        24,
        6,
    )
    manifest = p.read(evidence["out"] / "manifest.json")
    assert set(manifest["weights"]) == p.expected_names()
    with tarfile.open(evidence["out"] / p.ARCHIVE) as archive:
        assert set(archive.getnames()) == set(manifest["members"])
        for item in archive:
            prefix, relative = item.name.split("/", 1)
            assert archive.extractfile(item).read() == (evidence[prefix] / relative).read_bytes()
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "scripts/publish_chess_candidate.py"),
            "audit",
            "--publication",
            str(evidence["out"]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "verified"
    with pytest.raises(FileExistsError):
        publish(evidence)


@pytest.mark.parametrize("fault", ["fit", "diagnostic", "gate", "cache", "arena", "mean"])
def test_summary_rejects_missing_panels_or_false_aggregates(source, fault):
    plan, summary = p.read(source["plan"]), p.read(source["report"] / "summary.json")
    if fault == "fit":
        del summary["metrics"]["delta-127"]
    elif fault == "diagnostic":
        del summary["permutation_diagnostic"]["delta-127"]["shift"]
    elif fault == "gate":
        summary["continuation_passed"] = True
    elif fault == "cache":
        summary["training_cache"]["legal_candidates"] += 1
    elif fault == "arena":
        summary["arena"]["by_opponent"]["direct"]["delta"]["score_lower_bound"] += 0.1
    else:
        summary["means"]["delta"]["dev"]["agreement"] += 0.1
    with pytest.raises(ValueError):
        p.validate_summary(plan, summary)


@pytest.mark.parametrize(
    "member",
    ["training-cache.json", "delta-97/permuted-dev.jsonl", "arena/game-288.pgn", "direct-109/weights.pt"],
)
def test_removed_required_member_cannot_be_hidden_by_resealing(evidence, member):
    (evidence["execution"] / member).unlink()
    seal(
        evidence["execution"],
        evidence["report"],
        evidence["plan"],
        p.read(evidence["report"] / "summary.json"),
    )
    with pytest.raises(ValueError):
        publish(evidence)
    assert not evidence["out"].exists()


def test_source_report_must_reproduce_before_publication(evidence, monkeypatch):
    monkeypatch.setattr(
        p, "load_study", lambda plan=None: SimpleNamespace(verify_plan=p.read, report=lambda *_: {})
    )
    with pytest.raises(ValueError, match="reproduce"):
        publish(evidence)
    assert not evidence["out"].exists() and not evidence["models"].exists()


def test_diagnostic_offsets_and_cache_hashes_are_bound_to_receipts(evidence):
    path = evidence["execution"] / "delta-97/permuted-dev.jsonl"
    values = [json.loads(row) for row in path.read_text().splitlines()]
    values[0]["successor_permutation_offset"] = 0
    jsonl(path, values)
    receipt = p.read(path.parent / "completed.json")
    receipt["files"][path.name] = p.sha(path)
    write(path.parent / "completed.json", receipt)
    seal(
        evidence["execution"],
        evidence["report"],
        evidence["plan"],
        p.read(evidence["report"] / "summary.json"),
    )
    with pytest.raises(ValueError, match="Permutation"):
        publish(evidence)


def test_checkpoint_byte_tampering_fails_portable_audit(evidence):
    publish(evidence)
    with (evidence["models"] / "delta-97/weights.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        p.audit(evidence["out"])


def test_publisher_identity_is_checked_by_portable_audit(evidence):
    publish(evidence)
    path = evidence["out"] / "completed.json"
    receipt = p.read(path)
    receipt["publisher_sha256"] = "0" * 64
    write(path, receipt)
    with pytest.raises(ValueError, match="Audit implementation"):
        p.audit(evidence["out"])


@pytest.mark.parametrize("fault", ["order", "computation", "evaluation_time", "cache_bytes", "raw_game"])
def test_rehashed_semantic_corruption_is_rejected(evidence, fault):
    execution = evidence["execution"]
    summary = p.read(evidence["report"] / "summary.json")
    directory = execution / "delta-97"
    if fault in ("order", "computation"):
        learning = [json.loads(line) for line in (directory / "learning.jsonl").read_text().splitlines()]
        if fault == "order":
            learning[0]["indices_sha256"] = "1" * 64
        else:
            learning[0]["candidate_refinement_iterations"] += 1
        jsonl(directory / "learning.jsonl", learning)
        trained = p.read(directory / "training.json")
        trained["learning_sha256"] = p.sha(directory / "learning.jsonl")
        write(directory / "training.json", trained)
    if fault in ("order", "computation", "evaluation_time"):
        receipt = p.read(directory / "completed.json")
        if fault == "evaluation_time":
            receipt["evaluation"]["dev"]["evaluation_wall_seconds"] = -1
        receipt["files"] = {name: p.sha(directory / name) for name in receipt["files"]}
        write(directory / "completed.json", receipt)
    elif fault == "cache_bytes":
        cache = p.read(execution / "training-cache.json")
        cache["cpu_tensor_bytes"] += 4
        write(execution / "training-cache.json", cache)
        summary["training_cache"] = cache
    else:
        path = execution / "arena/game-001.json"
        game = p.read(path)
        game["white_arm"] = "direct"
        write(path, game)
        receipt = p.read(execution / "arena/completed.json")
        receipt["files"][path.name] = p.sha(path)
        write(execution / "arena/completed.json", receipt)
    seal(execution, evidence["report"], evidence["plan"], summary)
    with pytest.raises(ValueError):
        publish(evidence)
    assert not evidence["out"].exists()


def test_failed_execution_is_not_packaged(evidence):
    write(evidence["execution"] / "failed.json", {"status": "failed"})
    with pytest.raises(ValueError, match="Failed evidence"):
        publish(evidence)


def test_nonpositive_reference_stays_undefined_without_changing_diagnostic_status(source):
    plan, summary = p.read(source["plan"]), p.read(source["report"] / "summary.json")
    for split in p.SPLITS:
        for seed in p.SEEDS:
            summary["regret"][f"direct-{seed}"][split] = 0.0
            summary["regret"][f"delta-{seed}"][split] = -0.1
    compare(summary, plan)
    p.validate_summary(plan, summary)
    assert all(
        row["relative_reduction"] is None for row in summary["comparisons"] if row["comparator"] == "direct"
    )
    assert not summary["continuation_passed"]


def test_rebound_compressed_hash_cannot_hide_changed_raw_member(evidence, tmp_path):
    publish(evidence)
    archive = evidence["out"] / p.ARCHIVE
    changed = tmp_path / "changed.tar.gz"
    import io

    with tarfile.open(archive) as source_archive, tarfile.open(changed, "w:gz") as target:
        for item in source_archive:
            raw = source_archive.extractfile(item).read()
            if item.name == "execution/delta-97/weights.pt":
                raw = bytes([raw[0] ^ 1]) + raw[1:]
            target.addfile(item, io.BytesIO(raw))
    changed.replace(archive)
    manifest = p.read(evidence["out"] / "manifest.json")
    manifest["archive"].update(sha256=p.sha(archive), size=archive.stat().st_size)
    write(evidence["out"] / "manifest.json", manifest)
    receipt = p.read(evidence["out"] / "completed.json")
    receipt["manifest_sha256"] = p.sha(evidence["out"] / "manifest.json")
    write(evidence["out"] / "completed.json", receipt)
    with pytest.raises(ValueError, match="member hash"):
        p.audit(evidence["out"])


def test_figure_and_receipt_retain_all_seeds_and_failed_verdict(evidence, tmp_path):
    publish(evidence)
    outputs = p.render(evidence["out"], tmp_path / "figure")
    receipt = p.read(outputs["json"])
    assert receipt["verification"]["fits"] == 12
    assert all(receipt["plots"][ext] == p.sha(outputs[ext]) for ext in ("png", "svg"))
    text = Path(outputs["svg"]).read_text()
    assert "FAILED" in text and "Mapping corruption" in text
    for label in p.LABELS.values():
        assert label in text
    for seed in p.SEEDS:
        assert f"Seed {seed}" in text
    with pytest.raises(FileExistsError):
        p.render(evidence["out"], tmp_path / "figure")
