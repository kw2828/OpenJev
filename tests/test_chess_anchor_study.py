"""Synthetic integration and adversarial receipt checks, without a real engine."""

import copy
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import pytest
import torch

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_spatial_data import generate_data, validate_data

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = load_script("test_anchor_study_runner", "chess_anchor_study.py")
grader = load_script("test_anchor_secondary_grader", "chess_compute_study.py")
FULL_PROTOCOL = copy.deepcopy(study.PROTOCOL)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in values))


def toy_rows():
    board = chess.Board()
    result = []
    for index, move in enumerate(("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6")):
        result.append(
            {
                "id": f"train-{index}",
                "game_id": 0,
                "fen": board.fen(en_passant="fen"),
                "target_uci": min(m.uci() for m in board.legal_moves),
                "target_value": math.tanh(100 / 600),
            }
        )
        board.push_uci(move)
    return result


def test_default_paired_schedules_exact_batches_epochs_and_iteration_budget():
    for seed in (17, 29, 43):
        fixed = study.schedule(seed, "fixed")
        mixed = study.schedule(seed, "mixed")
        assert len(fixed) == len(mixed) == 1536
        assert [r["indices"] for r in fixed] == [r["indices"] for r in mixed]
        assert Counter(r["depth"] for r in fixed) == {4: 1536}
        assert Counter(r["depth"] for r in mixed) == {2: 512, 4: 512, 6: 512}
        assert sum(r["depth"] for r in fixed) == sum(r["depth"] for r in mixed) == 6144
        assert sum(len(r["indices"]) for r in mixed) == 32768 * 6
        for epoch in range(1, 7):
            assert sorted(i for r in mixed if r["epoch"] == epoch for i in r["indices"]) == list(range(32768))
        assert mixed == study.schedule(seed, "mixed")
    assert study.schedule(17, "fixed")[0]["indices"] != study.schedule(29, "fixed")[0]["indices"]
    configs = study.configs()
    assert len(configs) == 12
    assert {(c["arm"], c["seed"]) for c in configs} == {
        (arm, seed) for arm in study.ARMS for seed in (17, 29, 43)
    }
    assert configs == study.configs()


def test_same_seed_initialization_and_active_parameter_count():
    digests = set()
    for recurrence, _ in study.ARMS.values():
        model = AnchorChess(recurrence, 17)
        assert model.parameter_count() == 43726
        assert model.parameter_count() - sum(p.numel() for p in model.aux_head.parameters()) == 33185
        digests.add(study.digest_state(model))
    assert len(digests) == 1


@pytest.fixture
def signature_root(tmp_path, monkeypatch):
    root = tmp_path / "source"
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "SOURCES", ["sentinel.py"])
    for name in (
        "sentinel.py",
        study.ENGINE,
        "uv.lock",
        "pyproject.toml",
        "src/openjev/research/chess_spatial.py",
        "older-positions.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    write_rows(root / study.TRAIN, toy_rows())
    write(
        root / "runs/chess-spatial-v1/execution/data/completed.json",
        {"files": {"train.jsonl": study.sha(root / study.TRAIN)}},
    )
    write(
        root / study.OLD_PLAN,
        {
            "sources": {
                "src/openjev/research/chess_spatial.py": study.sha(
                    root / "src/openjev/research/chess_spatial.py"
                )
            }
        },
    )
    monkeypatch.setattr(
        study,
        "collect_exclusions",
        lambda _: {
            "states": [],
            "files": {"older-positions.json": study.sha(root / "older-positions.json")},
            "counts": {},
        },
    )
    return root


def test_prepare_freezes_inputs_without_training_and_refuses_overwrite(signature_root, tmp_path):
    plan = study.prepare(tmp_path / "plan")
    frozen = study.verify_plan(plan)
    assert frozen["active_parameters"] == 33185
    assert frozen["sources"] == {"sentinel.py": study.sha(signature_root / "sentinel.py")}
    assert study.read(plan.parent / "prepared.json")["training_or_scoring_started"] is False
    with pytest.raises(FileExistsError):
        study.prepare(plan.parent)


@pytest.mark.parametrize("changed", ["source", "train", "engine", "exclusion", "old_model", "plan"])
def test_frozen_signature_rejects_changed_bindings(signature_root, tmp_path, changed):
    plan = study.prepare(tmp_path / "plan")
    if changed == "plan":
        frozen = study.read(plan)
        frozen["protocol"]["width"] += 1
        write(plan, frozen)
    else:
        name = {
            "source": "sentinel.py",
            "train": study.TRAIN,
            "engine": study.ENGINE,
            "exclusion": "older-positions.json",
            "old_model": "src/openjev/research/chess_spatial.py",
        }[changed]
        with (signature_root / name).open("a") as stream:
            stream.write("changed\n")
    with pytest.raises(ValueError):
        study.verify_plan(plan)


@pytest.fixture
def gate_inputs(monkeypatch):
    monkeypatch.setitem(study.PROTOCOL, "bootstrap_replicates", 20)
    metrics, predictions, regret = {}, {}, {}
    positions = {
        split: [{"id": f"{split}-{i}", "game_id": i // 10} for i in range(100)] for split in ("dev", "shift")
    }
    counts = {
        ("residual_mixed", 4): 25,
        ("residual_mixed", 8): 30,
        ("anchor_mixed", 4): 35,
        ("anchor_mixed", 8): 45,
        ("residual_fixed", 4): 25,
        ("anchor_fixed", 4): 28,
    }
    for config in study.configs():
        name, arm = config["name"], config["arm"]
        metrics[name], predictions[name] = {}, {}
        for depth in study.PROTOCOL["evaluation_depths"]:
            count = counts.get((arm, depth), 20)
            for split in positions:
                key = f"{split}-d{depth}"
                metrics[name][key] = {"agreement": count / 100, "correct": count, "examples": 100}
                predictions[name][key] = [{"correct": i < count} for i in range(100)]
            if config["regime"] == "mixed" and depth in (4, 8):
                score = {
                    ("residual_mixed", 4): 0.3,
                    ("residual_mixed", 8): 0.25,
                    ("anchor_mixed", 4): 0.2,
                    ("anchor_mixed", 8): 0.1,
                }[arm, depth]
                regret[f"{name}-d{depth}"] = dict.fromkeys(positions, score)
    return metrics, predictions, regret, positions


def test_gate_all_48_configurations_and_predeclared_comparisons(gate_inputs):
    result = study.gate_and_comparisons(*gate_inputs)
    assert sum(len(value) for value in gate_inputs[0].values()) == 48 * 2
    assert result["primary_passed"] and result["extra_compute_passed"]
    assert len(result["comparisons"]) == 6 and len(result["checks"]) == 14
    assert set(result["means"]) == set(study.ARMS)
    for values in result["means"].values():
        assert set(values) == {"2", "4", "8", "16"}
        assert all(set(v) == {"dev", "shift"} for v in values.values())
    assert result["interaction_at_depth4"]["dev"] == pytest.approx(0.07)
    primary = result["comparisons"][0]
    assert primary["comparison"] == "primary"
    assert primary["game_cluster_interval"]["games"] == 10
    assert primary["game_cluster_interval"]["mean"] == pytest.approx(0.1)
    assert primary["bounded_regret_delta"] == pytest.approx(-0.1)


@pytest.mark.parametrize("failure", ["worst_seed", "engine_tie", "extra_depth"])
def test_gate_honors_failure_rules_and_cannot_select_depth16(gate_inputs, failure):
    metrics, predictions, regret, _ = gate_inputs
    if failure == "worst_seed":
        metrics["anchor_mixed-17"]["dev-d4"]["agreement"] = 0.2
        predictions["anchor_mixed-17"]["dev-d4"] = [{"correct": i < 20} for i in range(100)]
    elif failure == "engine_tie":
        for seed in study.PROTOCOL["seeds"]:
            regret[f"anchor_mixed-{seed}-d4"]["dev"] = regret[f"residual_mixed-{seed}-d4"]["dev"]
    else:
        for seed in study.PROTOCOL["seeds"]:
            name = f"anchor_mixed-{seed}"
            for depth, count in ((8, 30), (16, 100)):
                metrics[name][f"dev-d{depth}"]["agreement"] = count / 100
                predictions[name][f"dev-d{depth}"] = [{"correct": i < count} for i in range(100)]
    result = study.gate_and_comparisons(*gate_inputs)
    assert not result["extra_compute_passed" if failure == "extra_depth" else "primary_passed"]


@pytest.mark.parametrize("missing", ["model", "depth", "prediction", "extra", "grade"])
def test_gate_rejects_incomplete_or_extra_membership(gate_inputs, missing):
    metrics, predictions, regret, _ = gate_inputs
    if missing == "model":
        metrics.pop("anchor_fixed-43")
    elif missing == "depth":
        metrics["residual_fixed-17"].pop("shift-d16")
    elif missing == "prediction":
        predictions["residual_fixed-17"].pop("dev-d2")
    elif missing == "grade":
        regret.pop("anchor_mixed-17-d4")
    else:
        metrics["unplanned-99"] = copy.deepcopy(metrics["anchor_fixed-17"])
    with pytest.raises(ValueError):
        study.gate_and_comparisons(*gate_inputs)


@pytest.fixture
def tiny_execution(tmp_path, monkeypatch):
    root = tmp_path / "source"
    protocol = copy.deepcopy(FULL_PROTOCOL)
    protocol.update(
        seeds=[17],
        width=4,
        train_examples=6,
        epochs=1,
        batch_size=2,
        updates_per_fit=3,
        core_iterations_per_fit=12,
        training_device="cpu",
        regret_positions_per_split=2,
        latency_positions_per_split=1,
        bootstrap_replicates=20,
    )
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "PROTOCOL", protocol)
    write_rows(root / study.TRAIN, toy_rows())
    engine_path = root / study.ENGINE
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_text("Synthetic test engine, never executed\n")
    monkeypatch.setattr(study, "collect_exclusions", lambda _: {"states": [], "files": {}, "counts": {}})
    fresh = copy.deepcopy(study.FRESH_CONFIG)
    for split in fresh["splits"]:
        split.update(examples=4, game_cap=4, max_plies=8, random_move_probability=1.0)

    def teacher(board):
        return {
            "target_uci": min(m.uci() for m in board.legal_moves),
            "score_cp": 100,
            "mate": None,
            "target_value": math.tanh(100 / 600),
            "requested_nodes": 2000,
            "reported_nodes": 2001,
            "wall_seconds": 0.001,
        }

    monkeypatch.setattr(
        study, "generate_fresh", lambda out, _engine, excluded: generate_data(out, teacher, fresh, excluded)
    )
    monkeypatch.setattr(study, "validate_fresh", lambda out, excluded: validate_data(out, fresh, excluded))
    monkeypatch.setattr(
        study,
        "signature",
        lambda: {
            "protocol": protocol,
            "configurations": study.configs(),
            "engine_sha256": study.sha(engine_path),
        },
    )
    monkeypatch.setattr(study, "grading_module", lambda: grader)

    class SyntheticEngine:
        def __init__(self):
            self.id = {"name": "Stockfish 19 synthetic test"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def configure(self, _):
            pass

        def analyse(self, board, limit, info, root_moves=None):
            assert not board.move_stack and limit.nodes == 20000 and info
            move = root_moves[0] if root_moves else min(board.legal_moves, key=lambda m: m.uci())
            assert move in board.legal_moves
            score = chess.engine.Cp(80 if root_moves else 100)
            return {"pv": [move], "score": chess.engine.PovScore(score, board.turn), "nodes": 20001}

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _: SyntheticEngine())
    plan = study.prepare(tmp_path / "plan")
    execution = tmp_path / "execution"
    study.run(plan, execution)
    return plan, execution


def test_real_tiny_cpu_training_and_complete_report(tiny_execution, tmp_path):
    plan, execution = tiny_execution
    summary = study.report(plan, execution, tmp_path / "report")
    assert summary["status"] == "completed" and len(summary["metrics"]) == 4
    assert summary["novelty_established"] is False and summary["elo_estimate"] is None
    assert sum(c["updates"] for c in summary["costs"].values()) == 12
    assert sum(c["core_iterations"] for c in summary["costs"].values()) == 48
    initial_hashes = set()
    for config in study.configs():
        directory = execution / config["name"]
        trained = study.read(directory / "training.json")
        assert trained["examples_seen"] == 6 and trained["updates"] == 3
        assert trained["core_iterations"] == 12
        initial_hashes.add(trained["initial_state_sha256"])
        counts = Counter(row["depth"] for row in study.rows(directory / "learning.jsonl"))
        assert counts == ({4: 3} if config["regime"] == "fixed" else {2: 1, 4: 1, 6: 1})
        trained_model = AnchorChess.load(directory / "weights.pt", expected_plan_sha256=study.sha(plan))
        fresh_model = AnchorChess(config["recurrence"], 17, 4, 4)
        assert any(
            not torch.equal(value, fresh_model.state_dict()[key])
            for key, value in trained_model.state_dict().items()
            if key.startswith("encoder.")
        )
        assert all(
            torch.equal(value, fresh_model.state_dict()[key])
            for key, value in trained_model.state_dict().items()
            if key.startswith("aux_head.")
        )
    assert len(initial_hashes) == 1
    assert all(type(row["game_id"]) is int for row in study.rows(execution / "data/dev.jsonl"))
    assert all(type(row["score_cp"]) is int for row in study.rows(execution / "regret/analyses.jsonl"))


def rebind_fit(execution, name):
    directory = execution / name
    trained = study.read(directory / "training.json")
    trained["learning_sha256"] = study.sha(directory / "learning.jsonl")
    write(directory / "training.json", trained)
    receipt = study.read(directory / "completed.json")
    receipt["files"] = {filename: study.sha(directory / filename) for filename in receipt["files"]}
    write(directory / "completed.json", receipt)
    complete = study.read(execution / "completed.json")
    complete["fits"][name] = study.sha(directory / "completed.json")
    write(execution / "completed.json", complete)


@pytest.mark.parametrize(
    "corruption",
    [
        "epoch",
        "depth",
        "primary_metric",
        "missing_file",
        "latency_device",
        "latency_depth",
        "latency_threads",
        "warmup_count",
        "warmup_total",
        "warmup_illegal",
    ],
)
def test_report_rejects_rehashed_semantic_corruption(tiny_execution, tmp_path, corruption):
    plan, execution = tiny_execution
    name = study.configs()[0]["name"]
    directory = execution / name
    if corruption in ("epoch", "depth"):
        values = study.rows(directory / "learning.jsonl")
        values[0][corruption] += 1
        write_rows(directory / "learning.jsonl", values)
    elif corruption in ("primary_metric", "missing_file"):
        receipt = study.read(directory / "completed.json")
        if corruption == "primary_metric":
            receipt["evaluation"]["dev-d4"]["metrics"]["agreement"] += 0.1
        else:
            receipt["files"].pop("dev-d16.jsonl")
            (directory / "dev-d16.jsonl").unlink()
        write(directory / "completed.json", receipt)
    else:
        values = study.read(directory / "latency.json")
        row = values["4"]
        if corruption == "latency_device":
            row["device"] = "cuda"
        elif corruption == "latency_depth":
            row["depth"] = 8
        elif corruption == "latency_threads":
            row["torch_threads"] = 16
        elif corruption == "warmup_count":
            row["warmup_records"].pop()
        elif corruption == "warmup_total":
            row["warmup_wall_ms"] += 100
        else:
            row["warmup_records"][0]["choice"] = "a1a8"
        write(directory / "latency.json", values)
    rebind_fit(execution, name)
    with pytest.raises(ValueError):
        study.report(plan, execution, tmp_path / "corrupt-report")


@pytest.mark.parametrize(
    "invalid_seconds",
    [-1.0, math.nan, math.inf, -math.inf],
    ids=["negative", "nan", "positive_infinity", "negative_infinity"],
)
def test_report_rejects_rehashed_invalid_evaluation_time(tiny_execution, tmp_path, invalid_seconds):
    plan, execution = tiny_execution
    name = study.configs()[0]["name"]
    receipt_path = execution / name / "completed.json"
    receipt = study.read(receipt_path)
    receipt["evaluation"]["dev-d4"]["evaluation_wall_seconds"] = invalid_seconds
    # Deliberately bypass the strict writer to test hostile receipts accepted by
    # Python's JSON parser. Bind the changed receipt so rejection must be semantic.
    receipt_path.write_text(json.dumps(receipt, allow_nan=True) + "\n")
    complete = study.read(execution / "completed.json")
    complete["fits"][name] = study.sha(receipt_path)
    write(execution / "completed.json", complete)
    with pytest.raises(ValueError):
        study.report(plan, execution, tmp_path / "corrupt-time-report")
