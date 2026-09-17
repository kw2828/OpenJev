"""Small synthetic chess positions only; no benchmark files, fitting or engines."""

import copy
import json
import math

import chess
import pytest
import torch

from openjev.research import chessbench_eval as evaluation
from openjev.research.chess_candidate import ARMS, CandidateChess
from openjev.research.chess_spatial_data import state_key, symmetry_key


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def rows():
    start = chess.Board()
    ep = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        ep.push_uci(uci)
    castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    promotion = chess.Board("7k/8/8/8/8/8/p7/7K b - - 7 19")
    return [
        {
            "source_index": i * 11 + 1,
            "fen": board.fen(en_passant="fen"),
            "target_uci": target,
            "state_key": state_key(board),
            "symmetry_key": symmetry_key(board),
        }
        for i, (board, target) in enumerate(
            ((start, "e2e4"), (ep, "e5d6"), (castle, "e1g1"), (promotion, "a2a1n"))
        )
    ]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, records):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))


@pytest.mark.parametrize("arm", ARMS)
def test_batched_scores_match_native_single_position_policy_and_preserve_state(tmp_path, arm):
    model = CandidateChess(arm, seed=97, width=4)
    model.train()
    model.encoder.eval()  # Mixed module modes must survive the evaluator.
    before_modes = [module.training for module in model.modules()]
    before_weights = {k: v.clone() for k, v in model.state_dict().items()}
    before_rng = torch.random.get_rng_state().clone()
    path = tmp_path / "predictions.jsonl"
    source = rows()
    receipt = evaluation.evaluate(model, source, path, batch_size=3)
    assert receipt["status"] == "completed"
    assert receipt["batches"] == 2
    assert receipt["batch_size"] == 3
    assert receipt["model_state_unchanged"] is True
    assert receipt["device"] == "cpu" and receipt["torch_threads"] == 2
    assert "not single-decision latency" in receipt["timing_scope"]
    assert receipt["evaluation_wall_seconds"] >= 0
    assert receipt["metrics"]["positions"] == 4
    assert not any("value" in key for key in receipt["metrics"])
    assert [module.training for module in model.modules()] == before_modes
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert all(torch.equal(before_weights[k], v) for k, v in model.state_dict().items())
    predictions = read(path)
    total = sum(len(record["legal_ids"]) for record in predictions)
    assert receipt["candidate_evaluations"] == total
    assert receipt["root_encodings"] == len(source)
    for key in ("native_successors", "native_board_copies", "native_pushes", "successor_encodings"):
        assert receipt[key] == (total if arm in ("delta", "full_afterstate") else 0)
    assert evaluation.audit_predictions(path, source, expected_arm=arm) == receipt["metrics"]
    for record, row in zip(predictions, source, strict=True):
        choice = model.choose(chess.Board(row["fen"]), depth=4)
        assert record["choice"] == choice["choice"]
        assert record["target_probability"] == pytest.approx(
            choice["probabilities"][row["target_uci"]], rel=2e-6
        )
        assert record["confidence"] == pytest.approx(max(choice["probabilities"].values()), rel=2e-6)
        assert record["target_nll"] == pytest.approx(-math.log(record["target_probability"]), abs=1e-12)


@pytest.fixture(scope="module")
def saved_panel(tmp_path_factory):
    destination = tmp_path_factory.mktemp("synthetic-chessbench-eval") / "predictions.jsonl"
    source = rows()
    receipt = evaluation.evaluate(CandidateChess("delta", width=4), source, destination, batch_size=2)
    return source, destination, receipt


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "reordered",
        "duplicate",
        "fen",
        "state",
        "menu",
        "logit",
        "choice",
        "nll",
        "probability",
        "confidence",
        "arm",
        "model",
        "source_bool",
        "value",
    ],
)
def test_audit_rejects_tampering_without_calling_a_model(saved_panel, tmp_path, monkeypatch, fault):
    source, original, _ = saved_panel
    records = read(original)
    if fault == "missing":
        records.pop()
    elif fault == "reordered":
        records.reverse()
    elif fault == "duplicate":
        records[1] = copy.deepcopy(records[0])
    elif fault == "fen":
        records[0]["fen"] = source[1]["fen"]
    elif fault == "state":
        records[0]["state_key"] = source[1]["state_key"]
    elif fault == "menu":
        records[0]["legal_ids"].reverse()
    elif fault == "logit":
        records[0]["logits"][0] += 10
    elif fault == "choice":
        record = records[0]
        record["choice"] = next(uci for uci in record["legal_ids"] if uci != record["choice"])
    elif fault == "nll":
        records[0]["target_nll"] += 0.001
    elif fault == "probability":
        records[0]["target_probability"] += 0.001
    elif fault == "confidence":
        records[0]["confidence"] = -1e-15
    elif fault == "arm":
        records[0]["arm"] = "direct"
    elif fault == "model":
        records[1]["state_sha256"] = "0" * 64
    elif fault == "source_bool":
        records[0]["source_index"] = True
    else:
        records[0]["value_mae"] = 0.0
    path = tmp_path / "changed.jsonl"
    write(path, records)
    monkeypatch.setattr(CandidateChess, "forward", lambda *a, **k: pytest.fail("Audit called the model"))
    with pytest.raises(ValueError):
        evaluation.audit_predictions(path, source, expected_arm="delta")


def test_ties_choose_first_ordered_legal_id_and_extreme_logits_stay_finite(tmp_path):
    source = rows()[:1]
    model = CandidateChess("direct", width=4)
    with torch.no_grad():
        for parameter in model.policy_head.parameters():
            parameter.zero_()
    path = tmp_path / "ties.jsonl"
    receipt = evaluation.evaluate(model, source, path)
    record = read(path)[0]
    assert record["choice"] == min(chess.Board().legal_moves, key=lambda move: move.uci()).uci()
    assert record["target_nll"] == pytest.approx(math.log(20))
    assert receipt["metrics"]["mean_confidence"] == pytest.approx(1 / 20)
    record["logits"] = [1000.0] + [-1000.0] * 19
    record.update(evaluation._arithmetic(record["logits"], record["legal_ids"], record["target_uci"]))
    assert record["target_probability"] == 0.0 and math.isfinite(record["target_nll"])
    write(path, [record])
    assert evaluation.audit_predictions(path, source)["mean_nll"] > 1900


def test_existing_file_is_never_overwritten_or_evaluated(tmp_path, monkeypatch):
    path = tmp_path / "existing.jsonl"
    path.write_text("preserved partial attempt\n")
    monkeypatch.setattr(
        CandidateChess, "forward", lambda *a, **k: pytest.fail("Existing output was evaluated")
    )
    with pytest.raises(FileExistsError):
        evaluation.evaluate(CandidateChess("direct", width=4), rows(), path)
    assert path.read_text() == "preserved partial attempt\n"


@pytest.mark.parametrize("fault", ["nonfinite", "state_change"])
def test_forward_failure_keeps_output_and_restores_modes_and_threads(tmp_path, monkeypatch, fault):
    model = CandidateChess("direct", width=4)
    model.train()
    model.encoder.eval()
    modes = [module.training for module in model.modules()]
    original = model.forward

    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "nonfinite":
            result[0][0, 0] = torch.nan
        else:
            next(model.parameters()).add_(1)
        return result

    monkeypatch.setattr(model, "forward", broken)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    path = tmp_path / "failed.jsonl"
    try:
        with pytest.raises(ValueError, match="Nonfinite model output|Model state changed"):
            evaluation.evaluate(model, rows()[:1], path)
        assert torch.get_num_threads() == 1
        assert [module.training for module in model.modules()] == modes
        assert path.exists()
        with pytest.raises(FileExistsError):
            evaluation.evaluate(model, rows()[:1], path)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("fault", ["duplicate", "illegal", "identity", "value", "empty", "terminal"])
def test_invalid_inputs_fail_before_creating_output(tmp_path, fault):
    source = rows()
    if fault == "duplicate":
        source[1]["source_index"] = source[0]["source_index"]
    elif fault == "illegal":
        source[0]["target_uci"] = "e2e5"
    elif fault == "identity":
        source[0]["symmetry_key"] = "changed"
    elif fault == "value":
        source[0]["target_value"] = 0.0
    elif fault == "empty":
        source = []
    else:
        board = chess.Board("8/8/8/8/8/8/4k3/7K w - - 0 1")
        source = [
            {
                "source_index": 1,
                "fen": board.fen(),
                "target_uci": "h1g1",
                "state_key": state_key(board),
                "symmetry_key": symmetry_key(board),
            }
        ]
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError):
        evaluation.evaluate(CandidateChess("direct", width=4), source, path)
    assert not path.exists()


@pytest.mark.parametrize("fault", ["nan", "duplicate_key", "truncated"])
def test_bad_json_is_rejected(saved_panel, tmp_path, fault):
    source, original, _ = saved_panel
    raw = original.read_text()
    if fault == "nan":
        raw = raw.replace('"confidence": ', '"confidence": NaN, "unused": ', 1)
    elif fault == "duplicate_key":
        raw = raw.replace('"version": ', '"version": "duplicate", "version": ', 1)
    else:
        raw = raw.rstrip("\n")
    path = tmp_path / "bad-json.jsonl"
    path.write_text(raw)
    with pytest.raises(ValueError):
        evaluation.audit_predictions(path, source)


def test_incompatible_depth_and_non_float32_models_are_rejected(tmp_path):
    for model in (
        CandidateChess("direct", width=4, root_depth=2),
        CandidateChess("direct", width=4).double(),
    ):
        with pytest.raises(ValueError):
            evaluation.evaluate(model, rows(), tmp_path / "bad-model.jsonl")
