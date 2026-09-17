"""Tiny synthetic training and evidence contracts, not a chess benchmark."""

import copy
import hashlib
import json
import math
import random

import chess
import chess.engine
import numpy as np
import pytest
import torch

from openjev.research import chess_connectome_study as study
from openjev.research.chess_candidate import CandidateChess, encode_batch
from openjev.research.chess_connectome_adapter import ConnectomeChessAdapter
from openjev.research.connectome_graph import SignedGraph, graph_sha256

PLAN_SHA = "a" * 64
DATA_SHA = "c" * 64


@pytest.fixture(autouse=True)
def bounded_cpu_execution(monkeypatch):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)

    def no_real_engine(*args, **kwargs):
        raise AssertionError("These synthetic tests must never start a real engine")

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", no_real_engine)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def tiny_graph():
    return SignedGraph(
        node_ids=[10, 20, 30, 40, 50],
        groups=[0, 1, 2, 3, 0],
        sources=[0, 1, 2, 3, 4, 0, 3],
        destinations=[1, 2, 3, 4, 0, 3, 1],
        signs=[1, -1, 1, -1, 1, 1, -1],
        provenance={"fixture": "synthetic-only"},
    )


def tiny_model(*, mode="sparse", seed=17):
    return ConnectomeChessAdapter(
        CandidateChess("direct", width=4, seed=19), tiny_graph(),
        mode=mode, seed=seed, mapping_seed=23, slot_channels=1, steps=2,
    )


def synthetic_rows():
    fens = [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 12 20",
        "7k/8/8/8/3Pp3/8/8/7K b - d3 0 12",
        "7k/P7/8/8/8/8/8/7K w - - 7 19",
        "7k/6R1/8/5K2/8/8/8/8 b - - 0 1",
    ]
    return [
        {
            "id": f"synthetic-{index}",
            "game_id": index,
            "fen": fen,
            "target_uci": max(move.uci() for move in chess.Board(fen).legal_moves),
            "target_value": (-1) ** index * (0.1 + 0.1 * index),
        }
        for index, fen in enumerate(fens)
    ]


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state_equal(model, expected):
    assert model.state_dict().keys() == expected.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(variant="biological", seed=97):
    return next(copy.deepcopy(item) for item in study.configurations()
                if item["variant"] == variant and item["seed"] == seed)


def backbone():
    return CandidateChess("direct", width=4, seed=97)


def tiny_settings(**overrides):
    return study.TrainingSettings(
        **({"train_examples": 4, "epochs": 2, "batch_size": 2, "device": "cpu"} | overrides)
    )


@pytest.fixture
def private_runs(tmp_path, monkeypatch):
    directory = tmp_path / "runs"
    directory.mkdir()
    monkeypatch.setattr(study, "RUNS_ROOT", directory)
    return directory


def test_all_control_configurations_are_fixed_and_no_seed_is_selected():
    first = study.configurations()
    second = study.configurations()
    assert first == second
    assert len(first) == 18
    assert {(item["variant"], item["seed"]) for item in first} == {
        (variant, seed)
        for variant in ("biological", "rewire151", "rewire163", "rewire179", "dense", "node_local")
        for seed in (97, 109, 127)
    }


@pytest.mark.parametrize("overrides", [
    {"train_examples": 0}, {"train_examples": True}, {"epochs": 0}, {"epochs": True},
    {"batch_size": 0}, {"batch_size": True}, {"train_examples": 5, "batch_size": 2},
    {"learning_rate": 0}, {"learning_rate": float("nan")}, {"gradient_clip": 0},
    {"gradient_clip": float("inf")}, {"device": "unknown"},
])
def test_training_rejects_invalid_or_nonintegral_fixed_budgets(overrides):
    with pytest.raises((ValueError, TypeError)):
        tiny_settings(**overrides)


def test_schedule_exactly_covers_each_epoch_and_preserves_caller_rng():
    settings = tiny_settings()
    before = torch.random.get_rng_state().clone()
    first = study.schedule(97, settings)
    assert first == study.schedule(97, settings)
    assert len(first) == settings.updates == 4
    assert [step["step"] for step in first] == [1, 2, 3, 4]
    for epoch in (1, 2):
        batches = [step["indices"] for step in first if step["epoch"] == epoch]
        assert [len(batch) for batch in batches] == [2, 2]
        assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert torch.equal(torch.random.get_rng_state(), before)


def checkpoint_fixture(private_runs):
    graph, original, config = tiny_graph(), backbone(), configuration()
    model = study.make_model(config, original, graph)
    private_runs.mkdir(parents=True, exist_ok=True)
    path = private_runs / "weights.pt"
    digest = study.state_sha256(original)
    study.save_checkpoint(model, path, plan_sha256=PLAN_SHA,
                          configuration=config, backbone_sha256=digest)
    expected = {
        "expected_plan_sha256": PLAN_SHA,
        "expected_config": config,
        "expected_backbone_sha256": digest,
        "expected_checkpoint_sha256": file_sha(path),
    }
    return model, original, graph, config, path, expected


def test_checkpoint_roundtrip_matches_full_tensor_state_and_predictions(private_runs):
    model, original, graph, _, path, expected = checkpoint_fixture(private_runs)
    before = snapshot(original)
    restored = study.load_checkpoint(path, original, graph, **expected)
    assert_state_equal(restored, snapshot(model))
    assert_state_equal(original, before)
    batch, _ = encode_batch([chess.Board(row["fen"]) for row in synthetic_rows()[:2]], "direct")
    for actual, reference in zip(restored(**batch), model(**batch), strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert all(not value.requires_grad for value in restored.backbone.parameters())
    assert not restored.backbone.training


def test_checkpoint_save_never_overwrites_an_existing_attempt(private_runs):
    model, original, _, config, path, _ = checkpoint_fixture(private_runs)
    digest = file_sha(path)
    with pytest.raises(FileExistsError):
        study.save_checkpoint(model, path, plan_sha256=PLAN_SHA,
                              configuration=config, backbone_sha256=study.state_sha256(original))
    assert file_sha(path) == digest


@pytest.mark.parametrize("field", ["expected_plan_sha256", "expected_backbone_sha256",
                                   "expected_checkpoint_sha256", "expected_config"])
def test_checkpoint_load_requires_all_external_identity_bindings(private_runs, field):
    _, original, graph, _, path, expected = checkpoint_fixture(private_runs)
    expected[field] = configuration("dense") if field == "expected_config" else "f" * 64
    with pytest.raises((ValueError, TypeError)):
        study.load_checkpoint(path, original, graph, **expected)


def test_checkpoint_rejects_another_graph_or_backbone_even_when_file_hash_matches(private_runs):
    _, original, graph, _, path, expected = checkpoint_fixture(private_runs)
    changed_graph = SignedGraph(
        graph.node_ids, graph.groups, graph.destinations, graph.sources, graph.signs
    )
    assert graph_sha256(changed_graph) != graph_sha256(graph)
    with pytest.raises((ValueError, TypeError)):
        study.load_checkpoint(path, original, changed_graph, **expected)
    wrong_backbone = CandidateChess("direct", width=4, seed=109)
    with pytest.raises((ValueError, TypeError)):
        study.load_checkpoint(path, wrong_backbone, graph, **expected)


def test_checkpoint_rejects_truncated_file_without_resuming_or_replacing_it(private_runs):
    _, original, graph, _, path, expected = checkpoint_fixture(private_runs)
    truncated = private_runs / "truncated.pt"
    truncated.write_bytes(path.read_bytes()[:97])
    original_bytes = truncated.read_bytes()
    expected["expected_checkpoint_sha256"] = file_sha(truncated)
    with pytest.raises((ValueError, RuntimeError, EOFError)):
        study.load_checkpoint(truncated, original, graph, **expected)
    assert truncated.read_bytes() == original_bytes


@pytest.mark.parametrize("fault", [
    "version", "plan", "configuration", "architecture", "graph", "source", "state_hash",
    "extra_metadata", "missing_tensor", "extra_tensor", "tensor_shape", "tensor_dtype",
    "nonfinite", "backbone_value", "edge_identity", "edge_sign", "slot_mapping",
])
def test_checkpoint_internal_authentication_rejects_tampering_after_file_hash_update(private_runs, fault):
    _, original, graph, _, path, expected = checkpoint_fixture(private_runs)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    if fault == "version":
        payload["version"] = "unknown-version"
    elif fault == "plan":
        payload["plan_sha256"] = "e" * 64
    elif fault == "configuration":
        payload["configuration"] = configuration("dense")
    elif fault == "architecture":
        payload["architecture"]["graph_steps"] = 9
    elif fault == "graph":
        payload["graph_sha256"] = "e" * 64
    elif fault == "source":
        payload["source_sha256"] = "e" * 64
    elif fault == "state_hash":
        payload["state_sha256"] = "e" * 64
    elif fault == "extra_metadata":
        payload["optimizer_state"] = {}
    elif fault == "missing_tensor":
        del state["node_bias"]
    elif fault == "extra_tensor":
        state["unregistered_edge"] = torch.zeros(1)
    elif fault == "tensor_shape":
        state["node_bias"] = state["node_bias"][:1]
    elif fault == "tensor_dtype":
        state["node_bias"] = state["node_bias"].double()
    elif fault == "nonfinite":
        state["node_bias"][0] = float("nan")
    elif fault == "backbone_value":
        name = next(name for name in state if name.startswith("backbone.") and state[name].is_floating_point())
        state[name].view(-1)[0] += 1
    elif fault == "edge_identity":
        state["sources"][0] = (state["sources"][0] + 1) % 5
    elif fault == "edge_sign":
        state["signs"][0] *= -1
    elif fault == "slot_mapping":
        state["node_slots"][0] = (state["node_slots"][0] + 1) % 1024
    # Rebinding the transport checksum must not bypass internal expected identities.
    altered = private_runs / "altered.pt"
    torch.save(payload, altered)
    expected["expected_checkpoint_sha256"] = file_sha(altered)
    with pytest.raises((ValueError, TypeError)):
        study.load_checkpoint(altered, original, graph, **expected)


def test_two_tiny_fixed_budget_fits_repeat_exactly_and_never_change_the_backbone(private_runs):
    original, graph, config = backbone(), tiny_graph(), configuration()
    frozen = snapshot(original)
    rows = synthetic_rows()[:4]
    initial_rows = copy.deepcopy(rows)
    initial = study.make_model(config, original, graph)
    restored = []
    for index, caller_seed in enumerate((31, 987)):
        random.seed(caller_seed)
        np.random.seed(caller_seed)
        torch.manual_seed(caller_seed)
        out = private_runs / f"fit-{index}"
        result = study.fit(config, original, graph, rows, out, plan_sha256=PLAN_SHA,
                           data_sha256=DATA_SHA, settings=tiny_settings())
        assert result == json.loads((out / "training.json").read_text())
        learning = json_rows(out / "learning.jsonl")
        assert len(learning) == result["updates"] == 4
        assert result["examples_seen"] == 8
        assert result["backbone_unchanged"] is True
        assert result["data_sha256"] == DATA_SHA and result["plan_sha256"] == PLAN_SHA
        assert result["learning_sha256"] == file_sha(out / "learning.jsonl")
        assert result["checkpoint_sha256"] == file_sha(out / "weights.pt")
        assert result["actual_legal_candidates"] == 2 * sum(
            chess.Board(row["fen"]).legal_moves.count() for row in rows
        )
        cache = study.CachedPositions(rows, include_successors=False)
        assert study.audit_training(
            out, config, PLAN_SHA, DATA_SHA, cache.metadata, original, graph,
            settings=tiny_settings(), training_rows=rows,
        ) == result
        for record, step in zip(learning, study.schedule(config["seed"], tiny_settings()), strict=True):
            assert record["step"] == step["step"] and record["epoch"] == step["epoch"]
            assert record["indices_sha256"] == hashlib.sha256(
                json.dumps(step["indices"], separators=(",", ":")).encode()
            ).hexdigest()
            assert record["epoch_backbone_hash_checked"] == (step["step"] % 2 == 0)
            assert record["loss"] == pytest.approx(record["policy_ce"] + 0.5 * record["value_mse"], abs=1e-6)
            assert math.isfinite(record["gradient_norm_before_clip"])
            assert record["sampled_mps_allocated_bytes_after_step"] is None
        for field, value in result["computation"].items():
            assert value == sum(record["computation"][field] for record in learning)
        assert sorted(path.name for path in out.glob("*.pt")) == ["weights.pt"]
        restored.append(study.load_checkpoint(
            out / "weights.pt", original, graph,
            expected_plan_sha256=PLAN_SHA, expected_config=config,
            expected_backbone_sha256=study.state_sha256(original),
            expected_checkpoint_sha256=file_sha(out / "weights.pt"),
        ))
        assert_state_equal(original, frozen)
        assert_state_equal(restored[-1].backbone, frozen)
        assert rows == initial_rows
        assert not (out / "failed.json").exists()
    assert_state_equal(restored[0], snapshot(restored[1]))
    assert any(
        not torch.equal(value, initial.state_dict()[name])
        for name, value in restored[0].named_parameters() if value.requires_grad
    )


def test_fit_rejects_reuse_without_touching_existing_evidence(private_runs):
    out = private_runs / "existing-attempt"
    out.mkdir(parents=True)
    sentinel = out / "failed.json"
    sentinel.write_text('{"status":"failed","error":"synthetic prior attempt"}\n')
    before = sentinel.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        study.fit(configuration(), backbone(), tiny_graph(), synthetic_rows()[:4], out,
                  plan_sha256=PLAN_SHA, data_sha256=DATA_SHA, settings=tiny_settings())
    assert sentinel.read_bytes() == before
    assert sorted(path.name for path in out.iterdir()) == ["failed.json"]


def test_fit_failure_retains_partial_receipt_and_does_not_retry(private_runs, monkeypatch):
    original, graph, config = backbone(), tiny_graph(), configuration()
    before = snapshot(original)
    calls = []
    real_step = torch.optim.Adam.step

    def fail_second_step(optimizer, *args, **kwargs):
        calls.append(len(calls) + 1)
        if len(calls) == 2:
            raise RuntimeError("synthetic optimizer failure")
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", fail_second_step)
    out = private_runs / "failed-fit"
    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        study.fit(config, original, graph, synthetic_rows()[:4], out, plan_sha256=PLAN_SHA,
                  data_sha256=DATA_SHA, settings=tiny_settings())
    assert calls == [1, 2]
    assert json.loads((out / "failed.json").read_text())["status"] == "failed"
    assert not (out / "training.json").exists()
    assert not (out / "weights.pt").exists()
    assert_state_equal(original, before)
    first_attempt = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises((FileExistsError, ValueError)):
        study.fit(config, original, graph, synthetic_rows()[:4], out, plan_sha256=PLAN_SHA,
                  data_sha256=DATA_SHA, settings=tiny_settings())
    assert first_attempt == {path.name: path.read_bytes() for path in out.iterdir()}
    assert calls == [1, 2]


def test_graph_derived_checkpoint_cannot_be_written_outside_private_runs(private_runs, tmp_path):
    original, graph, config = backbone(), tiny_graph(), configuration()
    model = study.make_model(config, original, graph)
    forbidden = tmp_path / "public-weights.pt"
    with pytest.raises((ValueError, PermissionError)):
        study.save_checkpoint(model, forbidden, plan_sha256=PLAN_SHA, configuration=config,
                              backbone_sha256=study.state_sha256(original))
    assert not forbidden.exists()


def test_symlink_cannot_escape_private_checkpoint_boundary(private_runs, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (private_runs / "escape").symlink_to(outside, target_is_directory=True)
    original, graph, config = backbone(), tiny_graph(), configuration()
    with pytest.raises((ValueError, PermissionError)):
        study.save_checkpoint(study.make_model(config, original, graph), private_runs / "escape" / "weights.pt",
                              plan_sha256=PLAN_SHA, configuration=config,
                              backbone_sha256=study.state_sha256(original))
    assert not list(outside.iterdir())


@pytest.mark.parametrize("fault", ["row_count", "cache_tensor"])
def test_training_input_mismatch_fails_before_model_initialization(private_runs, monkeypatch, fault):
    rows = synthetic_rows()[:4]
    cache = study.CachedPositions(rows, include_successors=False)
    if fault == "row_count":
        inputs = rows[:3]
    else:
        cache.observations[0, 0, 0, 0] += 1
        inputs = cache

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid training data must fail before model initialization")

    monkeypatch.setattr(study, "make_model", forbidden)
    out = private_runs / "bad-input"
    with pytest.raises(ValueError):
        study.fit(configuration(), backbone(), tiny_graph(), inputs, out,
                  plan_sha256=PLAN_SHA, data_sha256=DATA_SHA, settings=tiny_settings())
    failed = json.loads((out / "failed.json").read_text())
    assert failed["updates_completed"] == 0
    assert not (out / "weights.pt").exists()


@pytest.mark.parametrize("fault", ["unavailable", "fallback_enabled"])
def test_mps_unavailable_or_fallback_does_not_silently_start_cpu_training(private_runs, monkeypatch, fault):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: fault == "fallback_enabled")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1" if fault == "fallback_enabled" else "0")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported MPS configuration must fail before creating a model")

    monkeypatch.setattr(study, "make_model", forbidden)
    out = private_runs / "bad-device"
    before = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    with pytest.raises(ValueError, match="MPS"):
        study.fit(configuration(), backbone(), tiny_graph(), synthetic_rows()[:4], out,
                  plan_sha256=PLAN_SHA, data_sha256=DATA_SHA, settings=tiny_settings(device="mps"))
    assert before == (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    assert json.loads((out / "failed.json").read_text())["updates_completed"] == 0
    assert not (out / "weights.pt").exists()


def evaluation_fixture(private_runs, variant="biological"):
    rows = synthetic_rows()
    if variant == "direct":
        config = copy.deepcopy(study.baseline_configurations()[0])
        model = backbone()
    else:
        config = configuration(variant)
        model = study.make_model(config, backbone(), tiny_graph())
        with torch.no_grad():
            model.output_projection.weight.fill_(0.125)
    cache = study.CachedPositions(rows, include_successors=False)
    path = private_runs / f"{variant}-predictions.jsonl"
    receipt = study.evaluate(model, cache, path, configuration=config, plan_sha256=PLAN_SHA, batch_size=2)
    return model, config, rows, cache, path, receipt


@pytest.mark.parametrize("variant", ["biological", "dense", "node_local", "direct"])
def test_evaluation_retains_all_legal_logits_order_and_independent_softmax_arithmetic(private_runs, variant):
    model, config, rows, cache, path, receipt = evaluation_fixture(private_runs, variant)
    saved = json_rows(path)
    expected_rows = []
    for start in range(0, len(rows), 2):
        batch_rows = rows[start:start + 2]
        inputs, menus = encode_batch([chess.Board(row["fen"]) for row in batch_rows], "direct")
        with torch.no_grad():
            logits, values, _ = model(**inputs)
        for offset, (row, menu) in enumerate(zip(batch_rows, menus, strict=True)):
            raw = logits[offset, :len(menu)].double()
            probabilities = raw.softmax(0)
            target = menu.index(row["target_uci"])
            expected_rows.append({
                "raw": raw.tolist(), "legal_ids": list(menu), "choice": menu[int(raw.argmax())],
                "nll": float(-raw.log_softmax(0)[target]), "probability": float(probabilities[target]),
                "value": float(values[offset]),
            })
    for position, expected, row in zip(saved, expected_rows, rows, strict=True):
        assert position["id"] == row["id"] and position["fen"] == row["fen"]
        assert position["legal_ids"] == expected["legal_ids"]
        assert position["logits"] == expected["raw"]
        assert position["choice"] == expected["choice"]
        assert position["target_nll"] == pytest.approx(expected["nll"], abs=1e-12)
        assert position["target_probability"] == pytest.approx(expected["probability"], abs=1e-12)
        assert position["value"] == expected["value"]
    assert receipt["metrics"]["examples"] == 5
    assert receipt["batches"] == 3
    assert receipt["model_state_unchanged"] is True
    assert receipt["actual_legal_candidates"] == sum(len(row["legal_ids"]) for row in saved)
    assert receipt["computation"]["native_successors"] == 0
    assert receipt["computation"]["frozen_root_iterations"] == 20
    assert receipt["predictions_sha256"] == file_sha(path)
    assert receipt["cache_sha256"] == cache.metadata["cache_sha256"]
    metrics, audited = study.audit_predictions(
        path, rows, expected_configuration=config, expected_plan_sha256=PLAN_SHA,
        expected_state_sha256=study.state_sha256(model),
    )
    assert metrics == receipt["metrics"] and audited == saved


def test_evaluation_audits_and_cost_reconstruction_need_no_forward_or_engine_calls(private_runs, monkeypatch):
    model, config, rows, cache, path, receipt = evaluation_fixture(private_runs)
    before = snapshot(model)

    def forbidden(*args, **kwargs):
        raise AssertionError("Auditing recorded evidence must never call the model")

    monkeypatch.setattr(model, "forward", forbidden)
    metrics, saved = study.audit_evaluation(
        receipt, path, rows, configuration=config, plan_sha256=PLAN_SHA,
        model=model, cache_metadata=cache.metadata,
    )
    assert metrics == receipt["metrics"] and saved == json_rows(path)
    assert_state_equal(model, before)


def test_evaluation_labels_do_not_enter_model_inputs_and_stable_ties_choose_first_legal_move(private_runs):
    config = copy.deepcopy(study.baseline_configurations()[0])
    model = backbone()
    with torch.no_grad():
        for value in model.policy_head.parameters():
            value.zero_()
    rows = synthetic_rows()
    changed = copy.deepcopy(rows)
    for row in changed:
        row["target_uci"] = min(move.uci() for move in chess.Board(row["fen"]).legal_moves)
        row["target_value"] = -0.25
    before = snapshot(model)
    first, second = private_runs / "targets-a.jsonl", private_runs / "targets-b.jsonl"
    study.evaluate(model, rows, first, configuration=config, plan_sha256=PLAN_SHA, batch_size=2)
    study.evaluate(model, changed, second, configuration=config, plan_sha256=PLAN_SHA, batch_size=2)
    for left, right in zip(json_rows(first), json_rows(second), strict=True):
        assert left["logits"] == right["logits"]
        assert left["legal_ids"] == right["legal_ids"]
        assert left["choice"] == right["choice"] == left["legal_ids"][0]
        assert left["value"] == right["value"]
        assert left["max_probability"] == pytest.approx(1 / len(left["legal_ids"]))
    assert_state_equal(model, before)


@pytest.mark.parametrize("fault", [
    "missing_row", "duplicate_row", "reordered_rows", "reordered_menu", "missing_logit",
    "nonfinite_logit", "wrong_choice", "nll", "probability", "correctness", "fen",
    "row_id", "game_id_type", "target_value", "configuration", "state_hash", "plan_hash",
])
def test_saved_prediction_corruption_is_rejected(private_runs, fault):
    model, config, rows, _, path, _ = evaluation_fixture(private_runs)
    saved = json_rows(path)
    if fault == "missing_row":
        saved.pop()
    elif fault == "duplicate_row":
        saved[1] = copy.deepcopy(saved[0])
    elif fault == "reordered_rows":
        saved[0], saved[1] = saved[1], saved[0]
    elif fault == "reordered_menu":
        saved[0]["legal_ids"].reverse()
        saved[0]["logits"].reverse()
    elif fault == "missing_logit":
        saved[0]["logits"].pop()
    elif fault == "nonfinite_logit":
        saved[0]["logits"][0] = float("inf")
    elif fault == "wrong_choice":
        saved[0]["choice"] = next(move for move in saved[0]["legal_ids"] if move != saved[0]["choice"])
    elif fault == "nll":
        saved[0]["target_nll"] += 0.1
    elif fault == "probability":
        saved[0]["target_probability"] *= 0.5
    elif fault == "correctness":
        saved[0]["correct"] = int(saved[0]["correct"])
    elif fault == "fen":
        saved[0]["fen"] = rows[1]["fen"]
    elif fault == "row_id":
        saved[0]["id"] = "other-row"
    elif fault == "game_id_type":
        saved[0]["game_id"] = str(saved[0]["game_id"])
    elif fault == "target_value":
        saved[0]["target_value"] += 0.1
    elif fault == "configuration":
        saved[0]["model_configuration"] = configuration("dense")
    elif fault == "state_hash":
        saved[0]["model_state_sha256"] = "d" * 64
    elif fault == "plan_hash":
        saved[0]["plan_sha256"] = "d" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in saved))
    with pytest.raises((ValueError, TypeError)):
        study.audit_predictions(path, rows, expected_configuration=config,
                                expected_plan_sha256=PLAN_SHA, expected_state_sha256=study.state_sha256(model))


@pytest.mark.parametrize("fault", ["count", "batch_count", "compute", "negative_time",
                                  "weights_changed", "prediction_hash", "metric"])
def test_evaluation_receipt_tampering_is_rejected(private_runs, fault):
    model, config, rows, cache, path, receipt = evaluation_fixture(private_runs)
    bad = copy.deepcopy(receipt)
    if fault == "count":
        bad["actual_legal_candidates"] += 1
    elif fault == "batch_count":
        bad["batches"] += 1
    elif fault == "compute":
        bad["computation"]["graph_dense_matmul_macs"] += 1
    elif fault == "negative_time":
        bad["evaluation_wall_seconds"] = -1
    elif fault == "weights_changed":
        bad["model_state_unchanged"] = False
    elif fault == "prediction_hash":
        bad["predictions_sha256"] = "d" * 64
    elif fault == "metric":
        bad["metrics"]["agreement"] += 0.1
    with pytest.raises(ValueError):
        study.audit_evaluation(bad, path, rows, configuration=config, plan_sha256=PLAN_SHA,
                               model=model, cache_metadata=cache.metadata)


def test_evaluation_overwrite_fails_before_a_second_forward(private_runs, monkeypatch):
    model, config, rows, _, path, _ = evaluation_fixture(private_runs)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing predictions must not be rescored")

    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(FileExistsError):
        study.evaluate(model, rows, path, configuration=config, plan_sha256=PLAN_SHA, batch_size=2)
    assert path.read_bytes() == before


def test_partial_evaluation_preserves_rows_restores_state_and_never_retries(private_runs, monkeypatch):
    config = configuration()
    model = study.make_model(config, backbone(), tiny_graph()).train()
    before = snapshot(model)
    real_forward = model.forward
    calls = []

    def fail_second(**inputs):
        calls.append(len(inputs["observations"]))
        if len(calls) == 2:
            raise RuntimeError("synthetic forward failure")
        return real_forward(**inputs)

    monkeypatch.setattr(model, "forward", fail_second)
    torch.set_num_threads(1)
    path = private_runs / "partial.jsonl"
    rows = synthetic_rows()
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        study.evaluate(model, rows, path, configuration=config, plan_sha256=PLAN_SHA, batch_size=2)
    assert calls == [2, 2]
    assert len(json_rows(path)) == 2
    assert model.training and not model.backbone.training
    assert torch.get_num_threads() == 1
    assert_state_equal(model, before)
    with pytest.raises(ValueError, match="coverage"):
        study.audit_predictions(path, rows, expected_configuration=config,
                                expected_plan_sha256=PLAN_SHA, expected_state_sha256=study.state_sha256(model))
    with pytest.raises(FileExistsError):
        study.evaluate(model, rows, path, configuration=config, plan_sha256=PLAN_SHA, batch_size=2)
    assert calls == [2, 2]


class SyntheticStockfish:
    """Protocol-faithful mock: never starts a process or supplies benchmark labels."""

    def __init__(self, fault=None):
        self.id = {"name": "Stockfish 19 synthetic test double"}
        self.calls = []
        self.configurations = []
        self.fault = fault

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def configure(self, values):
        self.configurations.append(copy.deepcopy(values))

    def analyse(self, board, limit, *, root_moves=None, **kwargs):
        self.calls.append((board.fen(), limit.nodes, None if root_moves is None else root_moves[0].uci()))
        if self.fault == "second_call" and len(self.calls) == 2:
            raise RuntimeError("synthetic engine failure")
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        chosen = moves[0] if root_moves is None else root_moves[0]
        cp = 100 if root_moves is None else (150 if chosen == moves[0] else -200)
        result = {
            "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
            "pv": [chosen], "nodes": limit.nodes + 3,
        }
        if self.fault == "illegal_pv":
            result["pv"] = [chess.Move.from_uci("a1a8")]
        elif self.fault == "missing_nodes":
            del result["nodes"]
        return result


def grading_fixture(private_runs, monkeypatch, *, fault=None):
    engine_file = private_runs / "synthetic-stockfish-marker"
    engine_file.write_bytes(b"Synthetic engine identity only. Never execute this file.\n")
    mock = SyntheticStockfish(fault)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: mock)
    first, second = synthetic_rows()[:2]
    names = [configuration()["name"], configuration("dense")["name"]]
    plan = {
        "engine_path": str(engine_file), "panels": {"dev": [first], "shift": [second]},
        "secondary_indices": {"dev": [0], "shift": [0]},
        "configurations": [{"id": name} for name in names],
    }
    decisions = []
    for split, row in (("dev", first), ("shift", second)):
        moves = sorted(move.uci() for move in chess.Board(row["fen"]).legal_moves)
        for index, name in enumerate(names):
            decisions.append({
                "configuration": name, "split": split, "panel_index": 0, "id": row["id"],
                "choice": moves[-1] if split == "shift" and index else moves[0],
            })
    return plan, decisions, mock, file_sha(engine_file)


def grade(plan, decisions, out, engine_sha):
    return study.score_stronger(plan, out, decisions, plan_sha256=PLAN_SHA,
                               expected_engine_sha256=engine_sha, call_ceiling=5, requested_node_ceiling=100000)


def audit_grade(plan, decisions, out, engine_sha):
    return study.audit_stronger(
        plan, out, decisions, expected_plan_sha256=PLAN_SHA,
        expected_engine_sha256=engine_sha, call_ceiling=5, requested_node_ceiling=100000,
    )


def test_stronger_grading_caches_duplicate_choices_and_reproduces_signed_losses(private_runs, monkeypatch):
    plan, decisions, mock, engine_sha = grading_fixture(private_runs, monkeypatch)
    out = private_runs / "grading"
    receipt = grade(plan, decisions, out, engine_sha)
    assert len(mock.calls) == receipt["cost"]["calls"] == 5
    assert receipt["cost"]["requested_nodes"] == 100000
    assert receipt["cost"]["reported_nodes"] == 100015
    assert all(call[1] == 20000 for call in mock.calls)
    assert sum(values == {"Clear Hash": None} for values in mock.configurations) == 5
    assert mock.configurations[0] == {"Threads": 1, "Hash": 16}
    analyses, records = json_rows(out / "analyses.jsonl"), json_rows(out / "regret.jsonl")
    assert len(analyses) == 5 and len(records) == 4
    assert receipt["cost"]["wall_seconds"] == sum(row["wall_seconds"] for row in analyses)
    for record in records:
        cp = -200 if record["split"] == "shift" and record["configuration"] == "dense-97" else 150
        assert record["cp_loss"] == 100 - cp
        assert record["bounded_regret"] == math.tanh(100 / 600) - math.tanh(cp / 600)
        mean = receipt["means"][record["split"]][record["configuration"]]
        assert mean["mean_cp_loss"] == record["cp_loss"]
        assert mean["mean_bounded_score_loss"] == record["bounded_regret"]
    assert any(row["bounded_regret"] < 0 for row in records)

    def forbidden(*args, **kwargs):
        raise AssertionError("Auditing engine receipts must never run an engine")

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", forbidden)
    assert audit_grade(plan, decisions, out, engine_sha) == receipt


@pytest.mark.parametrize("fault", ["calls", "nodes", "decision_missing", "decision_duplicate",
                                  "illegal_choice", "engine_hash"])
def test_grading_rejects_bad_coverage_or_budget_before_any_engine_call(private_runs, monkeypatch, fault):
    plan, decisions, mock, engine_sha = grading_fixture(private_runs, monkeypatch)
    kwargs = {"call_ceiling": 5, "requested_node_ceiling": 100000}
    if fault == "calls":
        kwargs["call_ceiling"] = 4
    elif fault == "nodes":
        kwargs["requested_node_ceiling"] = 99999
    elif fault == "decision_missing":
        decisions.pop()
    elif fault == "decision_duplicate":
        decisions[1] = copy.deepcopy(decisions[0])
    elif fault == "illegal_choice":
        decisions[0]["choice"] = "a1a8"
    elif fault == "engine_hash":
        engine_sha = "d" * 64
    with pytest.raises(ValueError):
        study.score_stronger(plan, private_runs / "invalid-grading", decisions,
                             plan_sha256=PLAN_SHA, expected_engine_sha256=engine_sha, **kwargs)
    assert mock.calls == []


@pytest.mark.parametrize("fault", ["second_call", "illegal_pv", "missing_nodes"])
def test_grading_failure_preserves_partial_attempt_without_retry(private_runs, monkeypatch, fault):
    plan, decisions, mock, engine_sha = grading_fixture(private_runs, monkeypatch, fault=fault)
    out = private_runs / "failed-grading"
    with pytest.raises((RuntimeError, ValueError)):
        grade(plan, decisions, out, engine_sha)
    failed = json.loads((out / "failed.json").read_text())
    assert failed["status"] == "failed"
    assert not (out / "completed.json").exists()
    calls = len(mock.calls)
    assert calls == (2 if fault == "second_call" else 1)
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(FileExistsError):
        grade(plan, decisions, out, engine_sha)
    assert len(mock.calls) == calls
    assert before == {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(ValueError):
        audit_grade(plan, decisions, out, engine_sha)


@pytest.mark.parametrize("fault", ["cp", "bounded_score", "requested_nodes", "reported_nodes_type",
                                  "pv", "duplicate_analysis", "missing_regret", "clipped_loss",
                                  "aggregate_cost", "mean", "engine_identity"])
def test_grading_audit_rejects_tampering_even_after_receipt_file_hashes_are_refreshed(
    private_runs, monkeypatch, fault
):
    plan, decisions, _, engine_sha = grading_fixture(private_runs, monkeypatch)
    out = private_runs / "tampered-grading"
    receipt = grade(plan, decisions, out, engine_sha)
    analyses, regret = json_rows(out / "analyses.jsonl"), json_rows(out / "regret.jsonl")
    if fault == "cp":
        analyses[0]["score_cp"] += 10
    elif fault == "bounded_score":
        analyses[0]["bounded_score"] += 0.1
    elif fault == "requested_nodes":
        analyses[0]["requested_nodes"] -= 1
    elif fault == "reported_nodes_type":
        analyses[0]["reported_nodes"] = True
    elif fault == "pv":
        analyses[0]["pv_first"] = "a1a8"
    elif fault == "duplicate_analysis":
        analyses[1] = copy.deepcopy(analyses[0])
    elif fault == "missing_regret":
        regret.pop()
    elif fault == "clipped_loss":
        assert regret[0]["bounded_regret"] < 0
        regret[0]["bounded_regret"] = 0
    elif fault == "aggregate_cost":
        receipt["cost"]["reported_nodes"] += 1
    elif fault == "mean":
        receipt["means"]["dev"]["biological-97"]["mean_bounded_score_loss"] += 0.1
    elif fault == "engine_identity":
        metadata = json.loads((out / "engine.json").read_text())
        metadata["id"]["name"] = "Other engine"
        (out / "engine.json").write_text(json.dumps(metadata))
    write_rows(out / "analyses.jsonl", analyses)
    write_rows(out / "regret.jsonl", regret)
    receipt["files"] = {name: file_sha(out / name) for name in receipt["files"]}
    (out / "completed.json").write_text(json.dumps(receipt))

    def forbidden(*args, **kwargs):
        raise AssertionError("A corrupted receipt must not trigger replacement engine calls")

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", forbidden)
    with pytest.raises(ValueError):
        audit_grade(plan, decisions, out, engine_sha)


def test_latency_retains_three_warmups_all_selected_calls_and_immutable_model(private_runs, monkeypatch):
    config, rows = configuration(), synthetic_rows()
    model = study.make_model(config, backbone(), tiny_graph()).train()
    before = snapshot(model)
    real_choose = model.choose
    calls = []

    def record(board):
        calls.append(board.fen())
        return real_choose(board)

    monkeypatch.setattr(model, "choose", record)
    selected = [3, 1]
    receipt = study.measure_latency(model, rows, selected, configuration=config, plan_sha256=PLAN_SHA)
    assert calls == [chess.STARTING_FEN] * 3 + [rows[index]["fen"] for index in selected]
    assert receipt["forward_calls"] == 5
    assert receipt["model_state_unchanged"] is True
    assert len(receipt["warmup_records"]) == 3 and len(receipt["records"]) == 2
    assert receipt["warmup_wall_ms"] == math.fsum(row["wall_ms"] for row in receipt["warmup_records"])
    assert receipt["total_wall_ms"] == math.fsum(row["wall_ms"] for row in receipt["records"])
    assert_state_equal(model, before)
    assert model.training and not model.backbone.training

    def forbidden(*args, **kwargs):
        raise AssertionError("Auditing recorded latency must never repeat a decision")

    monkeypatch.setattr(model, "choose", forbidden)
    assert study.audit_latency(
        receipt, rows, selected, configuration=config, plan_sha256=PLAN_SHA,
        expected_state_sha256=study.state_sha256(model), model=model,
    ) == receipt


@pytest.mark.parametrize("fault", ["warmup_count", "record_order", "total_time", "compute",
                                  "legal_menu", "probability", "choice", "internal_time"])
def test_latency_audit_rejects_coverage_arithmetic_and_decision_corruption(private_runs, fault):
    config, rows = configuration(), synthetic_rows()
    model = study.make_model(config, backbone(), tiny_graph())
    receipt = study.measure_latency(model, rows, [0, 1], configuration=config, plan_sha256=PLAN_SHA)
    if fault == "warmup_count":
        receipt["warmup_records"].pop()
    elif fault == "record_order":
        receipt["records"].reverse()
    elif fault == "total_time":
        receipt["total_wall_ms"] += 1
    elif fault == "compute":
        receipt["records"][0]["computation"]["graph_dense_matmul_macs"] += 1
    elif fault == "legal_menu":
        receipt["records"][0]["legal_ids"].reverse()
    elif fault == "probability":
        key = receipt["records"][0]["legal_ids"][0]
        receipt["records"][0]["decision"]["probabilities"][key] += 1
    elif fault == "choice":
        decision = receipt["records"][0]["decision"]
        decision["choice"] = next(key for key in decision["probabilities"] if key != decision["choice"])
    elif fault == "internal_time":
        receipt["records"][0]["decision"]["latency_ms"] = receipt["records"][0]["wall_ms"] + 1
    with pytest.raises(ValueError):
        study.audit_latency(receipt, rows, [0, 1], configuration=config, plan_sha256=PLAN_SHA,
                            expected_state_sha256=study.state_sha256(model), model=model)
