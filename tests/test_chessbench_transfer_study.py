"""Synthetic transfer orchestration checks; no benchmark decoding or neural forward."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
import torch

from openjev.research import chessbench_eval
from openjev.research.chess_spatial_data import state_key, symmetry_key

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def load_study():
    spec = importlib.util.spec_from_file_location(
        "synthetic_chessbench_transfer", ROOT / "scripts/chessbench_transfer_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_rows():
    board = chess.Board()
    result = []
    for index, move in enumerate(("e2e4", "e7e5")):
        result.append({"source_index": index, "fen": board.fen(en_passant="fen"),
                       "target_uci": move, "state_key": state_key(board),
                       "symmetry_key": symmetry_key(board)})
        board.push_uci(move)
    return result


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    study = load_study()
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "SOURCES", ["runner-source.py", "evaluator-source.py"])
    monkeypatch.setattr(study, "PROTOCOL", {**study.PROTOCOL, "positions": 2})
    monkeypatch.setattr(study, "environment", lambda: {"synthetic": True})
    monkeypatch.setattr(study.torch, "set_num_threads", lambda _: None)
    for name in [*study.SOURCES, "candidate-source.py"]:
        (tmp_path / name).write_text("# synthetic bytes\n")
    configs = [{"name": f"{arm}-{seed}", "arm": arm, "seed": seed, "width": 32}
               for arm in ("direct", "action_only", "delta", "full_afterstate")
               for seed in (97, 109, 127)]
    prior = {"states": [state_key(chess.Board())], "files": {"prior-file": "a" * 64},
             "counts": {"states": 1}}
    candidate = {
        "configurations": configs, "sources": {"candidate-source.py": study.sha(tmp_path / "candidate-source.py")},
        "exclusion_files": prior["files"], "exclusion_counts": prior["counts"],
        "exclusion_states_sha256": hashlib.sha256(json.dumps(prior["states"], separators=(",", ":")).encode()).hexdigest(),
        "arena_schedule": [{"id": "game-001"}],
    }
    save(tmp_path / study.CANDIDATE_PLAN, candidate)
    monkeypatch.setattr(study, "CANDIDATE_PLAN_SHA", study.sha(tmp_path / study.CANDIDATE_PLAN))
    original_execution = tmp_path / study.EXECUTION
    original_execution.mkdir(parents=True)
    save(original_execution / "data/analyses.jsonl", {"fen": chess.STARTING_FEN})
    save(original_execution / "arena/game-001.json", {
        "initial_fen": chess.STARTING_FEN, "final_fen": chess.STARTING_FEN,
        "moves": [], "attempts": [],
    })
    save(original_execution / "completed.json", {"status": "completed", "plan_sha256": study.CANDIDATE_PLAN_SHA,
                                                 "files": study.tree(original_execution)})
    publication = tmp_path / study.PUBLICATION
    models = tmp_path / "models/synthetic"
    weights = {}
    for config in configs:
        path = models / config["name"] / "weights.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(config["name"].encode())
        weights[config["name"]] = {"path": f"{config['name']}/weights.pt", "sha256": study.sha(path)}
    import os
    save(publication / "manifest.json", {
        "models_directory": os.path.relpath(models, publication), "weights": weights,
        "members": {f"execution/{name}": {"sha256": digest}
                    for name, digest in study.tree(original_execution).items()},
    })
    save(publication / "completed.json", {"status": "completed"})
    audit_calls = []

    def audit(path):
        audit_calls.append(path)
        if study.read(path / "completed.json")["status"] != "completed":
            raise ValueError("Synthetic publication incomplete")

    monkeypatch.setattr(study, "publisher", lambda: SimpleNamespace(audit=audit))
    monkeypatch.setattr(study.history, "collect_exclusions", lambda _: copy.deepcopy(prior))
    actual_collect = study.collect_exclusions
    monkeypatch.setattr(study, "collect_exclusions", lambda _: (prior["states"], {"total_natural_states": 1}))
    input_path = tmp_path / study.INPUT
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"opaque synthetic bytes; never decoded")
    monkeypatch.setattr(study, "INPUT_SHA", study.sha(input_path))
    save(tmp_path / study.DOWNLOAD, {"sha256": study.INPUT_SHA, "bytes": input_path.stat().st_size})
    selected = synthetic_rows()
    chosen = {"selected": selected, "rejection_log": [],
              "selection_receipt": {"status": "completed", "selected_count": len(selected), "shortfall": 0}}
    selection_calls, load_calls, forward_calls = [], [], []

    def selection(excluded):
        selection_calls.append(copy.deepcopy(excluded))
        return copy.deepcopy(chosen)

    monkeypatch.setattr(study, "selection", selection)

    def load(path, **expected):
        name = Path(path).parent.name
        config = next(c for c in configs if c["name"] == name)
        assert expected == {"expected_plan_sha256": study.CANDIDATE_PLAN_SHA,
                            "expected_arm": config["arm"], "expected_seed": config["seed"],
                            "expected_width": 32, "expected_root_depth": 4, "expected_branch_depth": 2}
        load_calls.append(name)
        tensor = torch.tensor([config["seed"], ["direct", "action_only", "delta", "full_afterstate"].index(config["arm"])],
                              dtype=torch.float32)
        return SimpleNamespace(**config, depth=4, branch_depth=2, state_dict=lambda: {"synthetic": tensor})

    monkeypatch.setattr(study, "CandidateChess", SimpleNamespace(load=load))

    def evaluate(model, selected_rows, path, batch_size):
        """Emit hand-constructed logits and the real receipt schema, without a forward."""
        forward_calls.append((model.name, copy.deepcopy(selected_rows), batch_size))
        identity = {"arm": model.arm, "seed": model.seed, "width": model.width,
                    "root_depth": 4, "branch_depth": 2, "state_sha256": chessbench_eval._state_sha(model)}
        predictions, candidates = [], 0
        for row in selected_rows:
            legal = sorted(m.uci() for m in chess.Board(row["fen"]).legal_moves)
            logits = [0.] * len(legal)
            logits[(model.seed + len(model.arm)) % len(logits)] = 1.
            predictions.append({"version": chessbench_eval.VERSION, **row, **identity, "legal_ids": legal,
                                "logits": logits, **chessbench_eval._arithmetic(logits, legal, row["target_uci"])})
            candidates += len(legal)
        study.write_rows(path, predictions)
        successor_count = candidates if model.arm in ("delta", "full_afterstate") else 0
        return {
            "status": "completed", "version": chessbench_eval.VERSION, "model": identity,
            "metrics": chessbench_eval._metrics(predictions), "input_rows_sha256": chessbench_eval._digest(selected_rows),
            "predictions_sha256": study.sha(path), "model_state_unchanged": True,
            "device": "cpu", "torch_threads": 2, "batch_size": batch_size,
            "batches": (len(selected_rows) + batch_size - 1) // batch_size, "evaluation_wall_seconds": .1,
            "timing_scope": chessbench_eval.TIMING_SCOPE, "root_encodings": len(selected_rows),
            "candidate_evaluations": candidates, "native_successors": successor_count,
            "native_board_copies": successor_count, "native_pushes": successor_count,
            "successor_encodings": successor_count,
            "probability_semantics": "Uncalibrated softmax over the complete native legal menu; not absolute move quality.",
        }

    monkeypatch.setattr(study.evaluation, "evaluate", evaluate)
    return SimpleNamespace(study=study, root=tmp_path, configs=configs, chosen=chosen, prior=prior,
                           audit_calls=audit_calls, selection_calls=selection_calls, load_calls=load_calls,
                           forward_calls=forward_calls, actual_collect=actual_collect, publication=publication,
                           original_execution=original_execution, monkeypatch=monkeypatch)


def prepared(fixture):
    return fixture.study.prepare(fixture.root / "protocol")


def executed(fixture):
    plan = prepared(fixture)
    execution = fixture.root / "execution"
    fixture.study.run(plan, execution)
    return plan, execution


def reseal(study, execution):
    receipt = study.read(execution / "completed.json")
    receipt["files"] = study.tree(execution)
    receipt["files"].pop("completed.json")
    save(execution / "completed.json", receipt)


def test_prerequisite_audits_publication_and_returns_all_twelve(fixture):
    plan, weights = fixture.study.prerequisite()
    assert fixture.audit_calls == [fixture.publication]
    assert list(weights) == [c["name"] for c in fixture.configs]
    assert len(weights) == 12 and plan["configurations"] == fixture.configs


@pytest.mark.parametrize("status", ["failed", "started"])
def test_prerequisite_rejects_incomplete_publication(fixture, status):
    save(fixture.publication / "completed.json", {"status": status})
    with pytest.raises(ValueError, match="publication incomplete"):
        fixture.study.prerequisite()


def test_prerequisite_requires_completion_receipt(fixture):
    (fixture.publication / "completed.json").unlink()
    with pytest.raises(FileNotFoundError):
        fixture.study.prerequisite()


def test_prerequisite_rejects_failed_candidate_even_when_files_resealed(fixture):
    save(fixture.original_execution / "failed.json", {"status": "failed"})
    reseal(fixture.study, fixture.original_execution)
    with pytest.raises(ValueError, match="incomplete or changed"):
        fixture.study.prerequisite()


@pytest.mark.parametrize("path", ["candidate-source.py", "models/synthetic/delta-109/weights.pt"])
def test_prerequisite_binds_sources_and_every_checkpoint(fixture, path):
    (fixture.root / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fixture.study.prerequisite()


def test_prerequisite_compares_local_raw_evidence_to_publication(fixture):
    save(fixture.original_execution / "data/analyses.jsonl", {"fen": "changed"})
    reseal(fixture.study, fixture.original_execution)
    with pytest.raises(ValueError, match="differs from its audited publication"):
        fixture.study.prerequisite()


def test_prepare_freezes_all_models_without_decoding_or_evaluation(fixture):
    plan_path = prepared(fixture)
    plan, exclusions = fixture.study.verify(plan_path)
    assert len(plan["configurations"]) == 12
    assert [c["name"] for c in plan["configurations"]] == [c["name"] for c in fixture.configs]
    assert exclusions == fixture.prior["states"]
    assert plan["decoded_benchmark_before_freeze"] is False
    assert not fixture.selection_calls and not fixture.load_calls and not fixture.forward_calls


def test_prepare_requires_matching_opaque_source(fixture):
    (fixture.root / fixture.study.INPUT).write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="Opaque source bytes"):
        prepared(fixture)
    assert not (fixture.root / "protocol").exists()


def test_prepare_exclusive_creation(fixture):
    path = prepared(fixture)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        prepared(fixture)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["source", "download", "input", "publication", "exclusion", "model_subset"])
def test_verify_rejects_frozen_binding_changes(fixture, change):
    study = fixture.study
    plan_path = prepared(fixture)
    if change == "source":
        (fixture.root / study.SOURCES[0]).write_text("changed")
    elif change == "download":
        save(fixture.root / study.DOWNLOAD, {"sha256": study.INPUT_SHA, "bytes": 1})
    elif change == "input":
        (fixture.root / study.INPUT).write_bytes(b"changed")
    elif change == "publication":
        save(fixture.publication / "completed.json", {"status": "completed", "changed": True})
    elif change == "exclusion":
        (plan_path.parent / "excluded-states.json.gz").write_bytes(b"changed")
    else:
        plan = study.read(plan_path)
        plan["configurations"].pop()
        save(plan_path, plan)
    with pytest.raises(ValueError):
        study.verify(plan_path)


def test_exclusions_include_native_successors_of_generator_and_arena(fixture):
    board = chess.Board()
    board.push_uci("e2e4")
    e4 = board.fen(en_passant="fen")
    board.push_uci("e7e5")
    e5 = board.fen(en_passant="fen")
    save(fixture.original_execution / "arena/game-001.json", {
        "initial_fen": chess.STARTING_FEN, "final_fen": e5,
        "moves": [{"fen_before": e4, "fen_after": e5}], "attempts": [{"fen_before": e5}],
    })
    candidate = fixture.study.read(fixture.root / fixture.study.CANDIDATE_PLAN)
    excluded, receipt = fixture.actual_collect(candidate)
    expected = set(fixture.prior["states"])
    visits = 0
    for fen in (chess.STARTING_FEN, e4, e5):
        position = chess.Board(fen)
        expected.add(state_key(position))
        for move in position.legal_moves:
            child = position.copy(stack=False)
            child.push(move)
            expected.add(state_key(child))
            visits += 1
    assert excluded == sorted(expected)
    assert receipt["candidate_distinct_roots"] == 3
    assert receipt["candidate_native_successor_visits"] == visits


def test_exclusions_require_historical_boundary_match(fixture):
    candidate = fixture.study.read(fixture.root / fixture.study.CANDIDATE_PLAN)
    candidate["exclusion_states_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Historical exposure boundary"):
        fixture.actual_collect(candidate)


def test_run_covers_all_twelve_same_selected_rows_and_depth_contract(fixture):
    _, execution = executed(fixture)
    assert fixture.load_calls == [c["name"] for c in fixture.configs]
    assert len(fixture.forward_calls) == 12
    assert all(rows == fixture.chosen["selected"] and batch == 16 for _, rows, batch in fixture.forward_calls)
    assert fixture.study.read(execution / "completed.json")["status"] == "completed"


def test_shortfall_preserves_partial_selection_and_does_not_evaluate(fixture):
    fixture.chosen["selected"] = fixture.chosen["selected"][:1]
    fixture.chosen["rejection_log"] = [{"synthetic": "rejected"}]
    fixture.chosen["selection_receipt"] = {"status": "failed", "selected_count": 1, "shortfall": 1}
    plan, execution = prepared(fixture), fixture.root / "execution"
    with pytest.raises(ValueError, match="quota not met"):
        fixture.study.run(plan, execution)
    assert fixture.study.rows(execution / "selected.jsonl") == fixture.chosen["selected"]
    assert fixture.study.rows(execution / "rejections.jsonl") == fixture.chosen["rejection_log"]
    assert fixture.study.read(execution / "failed.json")["status"] == "failed"
    assert not (execution / "completed.json").exists() and not fixture.load_calls


def test_evaluation_failure_keeps_partial_files_and_has_no_completion(fixture):
    original = fixture.study.evaluation.evaluate
    calls = []

    def fail_second(model, selected, path, batch_size):
        calls.append(model.name)
        if len(calls) == 2:
            path.write_text("partial output\n")
            raise RuntimeError("synthetic evaluation failure")
        return original(model, selected, path, batch_size)

    fixture.monkeypatch.setattr(fixture.study.evaluation, "evaluate", fail_second)
    plan, execution = prepared(fixture), fixture.root / "execution"
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        fixture.study.run(plan, execution)
    assert (execution / "direct-97.json").exists()
    assert (execution / "direct-109.jsonl").read_text() == "partial output\n"
    assert not (execution / "completed.json").exists()
    assert fixture.study.read(execution / "failed.json")["error"] == "synthetic evaluation failure"
    with pytest.raises(FileExistsError):
        fixture.study.run(plan, execution)
    assert len(calls) == 2


def test_report_reproduces_metrics_without_another_forward(fixture):
    plan, execution = executed(fixture)
    before = len(fixture.forward_calls)
    summary = fixture.study.report(plan, execution, fixture.root / "report")
    assert len(fixture.forward_calls) == before
    assert set(summary["metrics"]) == {c["name"] for c in fixture.configs}
    for arm in ("direct", "action_only", "delta", "full_afterstate"):
        values = [summary["metrics"][f"{arm}-{seed}"]["mean_nll"] for seed in (97, 109, 127)]
        assert summary["means"][arm]["mean_nll"] == pytest.approx(sum(values) / 3)
    assert len(summary["paired_differences"]) == 9
    assert summary["candidate_gates_changed"] is False and summary["elo_estimate"] is None


@pytest.mark.parametrize("kind", ["seed", "state", "cost", "time", "input_hash", "probability_hash", "metrics"])
def test_report_rejects_semantic_tamper_even_after_outer_hashes_resealed(fixture, kind):
    plan, execution = executed(fixture)
    study = fixture.study
    name = "delta-97"
    path = execution / f"{name}.json"
    receipt = study.read(path)
    if kind in ("seed", "state"):
        key, value = ("seed", 109) if kind == "seed" else ("state_sha256", "0" * 64)
        predictions = study.rows(execution / f"{name}.jsonl")
        for row in predictions:
            row[key] = value
        (execution / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
        digest = study.sha(execution / f"{name}.jsonl")
        receipt["predictions_sha256"] = digest
        receipt["evaluation"]["predictions_sha256"] = digest
        receipt["evaluation"]["model"][key] = value
    elif kind == "cost":
        receipt["evaluation"]["native_successors"] += 1
    elif kind == "time":
        receipt["evaluation"]["evaluation_wall_seconds"] = -1
    elif kind == "input_hash":
        receipt["evaluation"]["input_rows_sha256"] = "0" * 64
    elif kind == "probability_hash":
        receipt["evaluation"]["predictions_sha256"] = "0" * 64
    else:
        receipt["evaluation"]["metrics"]["agreement"] = 123
    save(path, receipt)
    reseal(study, execution)
    with pytest.raises(ValueError):
        study.report(plan, execution, fixture.root / "report")


def test_report_rejects_replaced_selection_even_after_file_reseal(fixture):
    plan, execution = executed(fixture)
    save(execution / "selection.json", {"status": "completed", "selected_count": 5000})
    reseal(fixture.study, execution)
    with pytest.raises(ValueError, match="Selection does not reproduce"):
        fixture.study.report(plan, execution, fixture.root / "report")


def test_report_preserves_failed_execution_as_failure(fixture):
    plan, execution = executed(fixture)
    save(execution / "failed.json", {"status": "failed"})
    reseal(fixture.study, execution)
    with pytest.raises(ValueError, match="incomplete or altered"):
        fixture.study.report(plan, execution, fixture.root / "report")


def test_tree_rejects_symlink(fixture):
    (fixture.root / "link").symlink_to(fixture.root / "runner-source.py")
    with pytest.raises(ValueError, match="Nonregular"):
        fixture.study.tree(fixture.root)
